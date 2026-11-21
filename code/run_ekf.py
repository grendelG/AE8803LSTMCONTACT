"""Replay a sequence through the Project 2b foothold EKF, optionally with per-foot reliability scores.

Every IMU sample -> predict. Two ways of feeding the feet (UPDATE_MODE):
- "touchdown"  : exactly project_2b/real_replay.py. Foot updates happen only at swing->stance events
                 (continuing feet are updated then too). A planted foot therefore gives body information
                 at most once or twice per stance, and in ~1/3 of stances never.
- "continuous" : the same events, PLUS an update of every planted foot whenever UPDATE_HZ worth of time
                 has passed since the last update. This is how leg-inertial EKFs are normally run; a slipping
                 foot now drags the body estimate continuously, so per-foot trust can actually matter.
                 Settings (UPDATE_HZ, SIGMA_MEAS, SIG_MIN) were chosen on the training sequences only
                 (tune_ekf.py -> results/ekf_tuning.json).
Reliability scores scale each foot's foothold random-walk sigma (log-linear map):
    sigma_f,i = SIG_MIN^(s_i) * SIG_MAX^(1 - s_i)
"""
from __future__ import annotations
import numpy as np, json, sys
import gtsam
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from foothold_filter import FootholdNavStateImuEKF
from common import *

CM = 0.01
FOOT_DT = 0.0           # feet stream shift in the replay; Project 2b timing. Calibrated feet->IMU shift is +5 ms,
                        # i.e. ~0; evaluation perturbs it by -40..+40 ms (experiment.OFFSETS)
# replay configuration; defaults = the tuned continuous replay (see tune_ekf.py)
UPDATE_MODE = "continuous"
UPDATE_HZ = 50.0
SIGMA_MEAS = 1 * CM     # foot-position measurement noise
SIG_MIN = 1e-4          # foothold random walk of a fully trusted foot (Project 2b's foothold_process_sigma)
TOUCHDOWN_CFG = dict(UPDATE_MODE="touchdown", SIGMA_MEAS=1 * CM, SIG_MIN=1e-4)   # original Project 2b replay
_tuned = ROOT / "results" / "ekf_tuning.json"
if _tuned.exists():
    _c = json.load(open(_tuned))["chosen"]
    UPDATE_HZ, SIGMA_MEAS, SIG_MIN = _c["UPDATE_HZ"], _c["SIGMA_MEAS"], _c["SIG_MIN"]
CONTINUOUS_CFG = dict(UPDATE_MODE="continuous", UPDATE_HZ=UPDATE_HZ, SIGMA_MEAS=SIGMA_MEAS, SIG_MIN=SIG_MIN)  # the default


def configure(**kw):
    """Set module-level replay settings, e.g. configure(**TOUCHDOWN_CFG). Returns the previous values.
    Settings persist in the process: callers that reuse processes (multiprocessing pools) must set them per job."""
    g = globals(); old = {k: g[k] for k in kw}
    g.update(kw)
    return old


class ReliabilityEKF(FootholdNavStateImuEKF):
    """Same filter; predict() takes one foothold process sigma per foot."""

    def predict(self, omega_body, specific_force_body, dt, foot_sigmas=None):
        if foot_sigmas is None:
            return super().predict(omega_body, specific_force_body, dt)
        self.x, _, U = self.predict_mean(self.gravity, self.x, omega_body, specific_force_body, dt)
        A_body = self.body_transition_jacobian(U, dt)
        dim = self.state_dim()
        F = np.eye(dim); F[0:9, 0:9] = A_body
        Q = np.zeros((dim, dim))
        Qb = np.zeros((9, 9))
        Qb[0:3, 0:3] = self.sigma_gyro ** 2 * np.eye(3)
        Qb[3:6, 3:6] = self.sigma_integration ** 2 * np.eye(3)
        Qb[6:9, 6:9] = self.sigma_acc ** 2 * np.eye(3)
        Q[0:9, 0:9] = Qb * dt
        for i in range(self.num_feet):
            s = self.foot_block(i)
            Q[s, s] = foot_sigmas[i] ** 2 * max(dt, 1e-9) * np.eye(3)
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x


def make_ekf():
    x0 = gtsam.NavState(gtsam.Rot3(), np.zeros(3), np.zeros(3))
    P0 = np.diag(np.concatenate([[1e-2, 1e-2, 1e-6], [0.05] * 3, [0.05] * 3, [0.08] * 12]))
    return ReliabilityEKF(x0=x0, P0=P0, num_feet=4, foot_names=FEET, sigma_gyro=8e-4, sigma_acc=1e-2,
                          foothold_process_sigma=SIG_MIN, sigma_meas=SIGMA_MEAS, sigma_height_prior=15 * CM)


def weighted_contact_update(ekf, old, new, r_scale, gate):
    """Same order as process_contact_measurements, but per-foot R = R0 * r_scale[foot];
    continuing feet with r_scale >= gate are skipped (gate=None disables)."""
    R0 = (ekf.sigma_meas ** 2) * np.eye(3)
    for foot in sorted(old):
        if gate is not None and r_scale[foot] >= gate:
            continue
        ekf.update_foothold_measurement(foot, old[foot], R_meas=R0 * r_scale[foot])
    for foot in sorted(new):
        ekf.reinitialize_foothold(foot, new[foot])
        ekf.update_foothold_measurement(foot, new[foot], R_meas=R0)


def run(seq, scores=None, sig_max=None, fix_timing=True, height_prior=False, mode="Q", r_max=100.0, gate=None, d_override=None):
    """scores: (N_imu, 4) reliability in [0,1] on the IMU time grid, or None for the fixed baseline."""
    d = load_raw(seq) if d_override is None else d_override
    t_imu = d["t_imu"]
    t_ft = d["t_ft"] + (FOOT_DT if fix_timing else 0.0)
    craw = d["contact_raw"]; pf = d["pfoot"]
    # merged, time-ordered events (IMU first on ties, as in a stable sort by time then order)
    ev_t = np.concatenate([t_imu, t_ft])
    ev_k = np.concatenate([np.zeros(len(t_imu), int), np.ones(len(t_ft), int)])
    ev_i = np.concatenate([np.arange(len(t_imu)), np.arange(len(t_ft))])
    order = np.lexsort((ev_k, ev_t))
    ekf = make_ekf()
    last_t = None; has_imu = False; omega = np.zeros(3); f = np.zeros(3)
    prev = None; last_emit = None; cur_imu = 0
    T, P, Rq = [], [], []

    def step(tnow):
        nonlocal last_t
        dt = 0.0 if last_t is None else max(tnow - last_t, 0.0)
        if dt > 0:
            if scores is None or mode != "Q":
                ekf.predict(omega, f, dt)
            else:
                s = np.nan_to_num(scores[cur_imu], nan=1.0).clip(0, 1)
                ekf.predict(omega, f, dt, foot_sigmas=SIG_MIN ** s * sig_max ** (1 - s))
        last_t = tnow

    for j in order:
        tt = ev_t[j]
        if ev_k[j] == 0:
            k = ev_i[j]; cur_imu = k
            omega = d["gyro"][k]; f = d["acc"][k]; has_imu = True
            step(tt)
            T.append(tt); P.append(np.array(ekf.x.position())); Rq.append(ekf.x.attitude().toQuaternion().coeffs())
        else:
            k = ev_i[j]
            cur = craw[k] == 1
            if prev is not None and has_imu:
                start = cur & ~prev
                cont = cur & prev
                due = (UPDATE_MODE == "continuous" and cont.any()
                       and (last_emit is None or tt - last_emit >= 1.0 / UPDATE_HZ - 1e-9))
                if (start.any() or due) and not (last_emit is not None and abs(tt - last_emit) < 1e-12):
                    step(tt)
                    old = {FEET[i]: pf[k, i].copy() for i in range(4) if cont[i]}
                    new = {FEET[i]: pf[k, i].copy() for i in range(4) if start[i]}
                    if scores is None or mode == "Q":
                        ekf.process_contact_measurements(old, new, z_world=0.0, apply_height_prior=height_prior)
                    else:
                        sc = np.nan_to_num(scores[cur_imu], nan=1.0).clip(0, 1)
                        # R mode: variance multiplier r_max^(1-s);  gate mode: drop feet with s < 0.5
                        rs = {FEET[i]: (r_max ** (1 - sc[i]) if mode == "R" else (1.0 if sc[i] >= 0.5 else 1e9))
                              for i in range(4)}
                        weighted_contact_update(ekf, old, new, rs, gate=(1e8 if mode == "gate" else None))
                    last_emit = tt
            prev = cur
    return np.array(T), np.array(P), np.array(Rq)   # quaternion (x,y,z,w)


if __name__ == "__main__":
    import time
    t0 = time.time()
    T, P, R = run(sys.argv[1] if len(sys.argv) > 1 else "building")
    print(len(T), P[-1], f"{time.time()-t0:.1f}s")
