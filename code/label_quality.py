"""Label quality: legacy labels (v1: LIO only, 10 Hz intervals, feet +0.06 s) vs current labels
(fused LIO+IMU reference, sliding 0.1 s window, calibrated feet timing).

1. Noise floor on clean data: how far does a planted foot "move" (median, tail), and how much of it is shared
   by the planted feet (a per-foot slip cannot be shared; reference error is).
2. Known-answer test: inject the synthetic slips of synth.py into the real data, recompute labels on the
   corrupted data, and check how well they recover the true injected slip motion over the same window.
Writes results/label_quality.json.
"""
import numpy as np, json
from common import *
from labels import CAL, stance_displacements, window_displacements
from synth import inject
from features import dedupe, SIGMA_LABEL


def auc(score, pos):
    o = np.argsort(score); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def pair_corr(N):
    """Correlation of displacement magnitudes between feet labelled in the same window."""
    a, b = [], []
    for k in np.where((~np.isnan(N)).sum(1) >= 2)[0]:
        idx = np.where(~np.isnan(N[k]))[0]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                a.append(N[k, idx[i]]); b.append(N[k, idx[j]])
    return float(np.corrcoef(a, b)[0, 1])


def legacy(d, dsy, delta):
    _, _, N = stance_displacements(d, CAL, 0.06)
    tg0, tg1, Ns = stance_displacements(dsy, CAL, 0.06)
    tf, dl = dedupe(d["t_ft"] + 0.06, delta)
    true = np.linalg.norm(interp(tf, dl, np.clip(tg1, tf[0], tf[-1])) - interp(tf, dl, np.clip(tg0, tf[0], tf[-1])), axis=2)
    return N, Ns, true


def current(d, dsy, delta):
    t, D = window_displacements(d, CAL, "fused")
    _, Ds = window_displacements(dsy, CAL, "fused")
    tf, dl = dedupe(d["t_ft"] + CAL["feet_dt"], delta)
    f = lambda tt: interp(tf, dl, np.clip(tt, tf[0], tf[-1]))
    true = np.linalg.norm(f(t + 0.05) - f(t - 0.05), axis=2)
    return np.linalg.norm(D, axis=2), np.linalg.norm(Ds, axis=2), true


if __name__ == "__main__":
    rep = {}
    for name, fn in [("legacy", legacy), ("current", current)]:
        clean, est, tru, pc = [], [], [], []
        for s in SEQS:
            d = load_raw(s)
            dsy, delta, _ = inject(d, 1000 + SEQS.index(s))
            N, Ns, true = fn(d, dsy, delta)
            clean.append(N[~np.isnan(N)]); pc.append(pair_corr(N))
            m = ~np.isnan(Ns); est.append(Ns[m]); tru.append(true[m])
        c, e, t = np.concatenate(clean), np.concatenate(est), np.concatenate(tru)
        score = lambda x: np.exp(-x ** 2 / (2 * SIGMA_LABEL ** 2))
        clean_slip = t < 0.005                      # windows with (almost) no injected slip motion
        rep[name] = dict(
            clean_median_cm=float(np.median(c) * 100), clean_p99_cm=float(np.percentile(c, 99) * 100),
            clean_frac_score_below_0p5=float(np.mean(score(c) < 0.5)),
            pair_magnitude_corr=float(np.mean(pc)),
            synth_corr_displacement=float(np.corrcoef(e, t)[0, 1]),
            synth_corr_score=float(np.corrcoef(score(e), score(t))[0, 1]),
            synth_AUC_slip_over_2cm=auc(e, t > 0.02),
            synth_false_alarm_rate=float(np.mean(score(e[clean_slip]) < 0.5)),
            synth_hit_rate=float(np.mean(score(e[t > 0.02]) < 0.5)),
        )
        print(name, {k: round(v, 3) for k, v in rep[name].items()}, flush=True)
    json.dump(rep, open(ROOT / "results" / "label_quality.json", "w"), indent=1)
