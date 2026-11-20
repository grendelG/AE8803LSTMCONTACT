"""Train the three reliability predictors and write per-sequence score files.

heuristic : rule-based, thresholds grid-searched on the training split
mlp       : current-timestep features -> score
lstm      : last 0.5 s (50 IMU steps) of the SAME features -> LSTM -> MLP -> score
Train = first 80% of building/under/garage/narrow; validation = last 20% of each; test = stair, trail.
All scores are 1.0 for swing feet (the EKF re-initialises footholds at touchdown anyway).
"""
import numpy as np, json, time, sys, os
import torch, torch.nn as nn
from common import *
from features import FEATURE_NAMES

torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)
F = len(FEATURE_NAMES)
FI = {n: i for i, n in enumerate(FEATURE_NAMES)}
WIN = 50
VAL_FRAC = 0.2
EXP = os.environ.get("EXP", "real")          # "real" (LIO labels) or "synth" (injected slips)
FEAT = ROOT / ("features" if EXP == "real" else "features_synth")
TAG = "" if EXP == "real" else "_synth"
OUT = ROOT / "results" / f"scores{TAG}"; OUT.mkdir(parents=True, exist_ok=True)


def load(seq):
    r = np.load(FEAT / f"{seq}.npz")
    return r["X"], r["y"]


def split_idx(N):
    cut = int(N * (1 - VAL_FRAC))
    return np.arange(WIN, cut), np.arange(cut + WIN, N)     # keep a window gap so val windows don't see train


# ---------------- heuristic ----------------
def heuristic(X, p):
    dev = X[..., FI["dev_norm"]]; tsd = X[..., FI["t_since_td"]]; ns = X[..., FI["n_stance"]]
    s = np.exp(-dev ** 2 / (2 * p["tau"] ** 2))
    s = s * (p["a"] + (1 - p["a"]) * np.minimum(1.0, tsd / p["T"]))
    s = s * np.where(ns <= 2, p["b"], 1.0)
    return np.where(X[..., FI["contact"]] > 0.5, s, 1.0)


def fit_heuristic(train):
    best = None
    for tau in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]:
        for a in [0.3, 0.5, 0.7, 0.85, 1.0]:
            for T in [0.03, 0.06, 0.1, 0.2]:
                for b in [0.6, 0.75, 0.9, 1.0]:
                    p = dict(tau=tau, a=a, T=T, b=b)
                    err = []; n = 0
                    for X, y, idx in train:
                        yy = y[idx]; m = ~np.isnan(yy)
                        err.append(np.sum((heuristic(X[idx], p)[m] - yy[m]) ** 2)); n += m.sum()
                    mse = sum(err) / n
                    if best is None or mse < best[0]:
                        best = (mse, p)
    return best[1]


# ---------------- networks ----------------
class MLP(nn.Module):
    def __init__(s, h=64):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(F, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(s, x):            # x (B, F) or (B, W, F) -> use last step
        if x.dim() == 3:
            x = x[:, -1]
        return torch.sigmoid(s.net(x)).squeeze(-1)


class LSTMMLP(nn.Module):
    def __init__(s, h=64):
        super().__init__()
        s.lstm = nn.LSTM(F, h, batch_first=True)
        s.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(s, x):            # (B, W, F)
        o, _ = s.lstm(x)
        return torch.sigmoid(s.head(o[:, -1])).squeeze(-1)


def make_samples(data, which):
    """List of (seq_id, t_index, foot) for labelled stance steps."""
    S = []
    for sid, (X, y, tr, va) in enumerate(data):
        idx = tr if which == "train" else va
        yy = y[idx]
        k, f = np.nonzero(~np.isnan(yy))
        S.append(np.column_stack([np.full(len(k), sid), idx[k], f]))
    return np.vstack(S)


def batch(datan, S, win):
    Xs = []
    offs = np.arange(-win + 1, 1)
    for sid in np.unique(S[:, 0]):
        m = S[:, 0] == sid
        Xn = datan[sid]
        ti = S[m, 1][:, None] + offs[None, :]
        Xs.append((m, Xn[ti, S[m, 2][:, None]]))
    out = np.zeros((len(S), win, F), np.float32)
    for m, x in Xs:
        out[m] = x
    return torch.from_numpy(out)


def train_net(kind, data, datan, mu, sd, epochs=12, bs=512, max_train=120000):
    win = WIN if kind == "lstm" else 1
    net = LSTMMLP() if kind == "lstm" else MLP()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    Str, Sva = make_samples(data, "train"), make_samples(data, "val")
    ytr = np.array([data[s][1][t, f] for s, t, f in Str], np.float32)
    yva = torch.tensor([data[s][1][t, f] for s, t, f in Sva])
    Xva = batch(datan, Sva, win)
    best = (1e9, None); log = []
    for ep in range(epochs):
        net.train(); t0 = time.time()
        perm = np.random.permutation(len(Str))[:max_train]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = batch(datan, Str[b], win); yb = torch.from_numpy(ytr[b])
            loss = ((net(xb) - yb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(Xva[i:i + 4096]) for i in range(0, len(Xva), 4096)])
            vl = float(((pv - yva) ** 2).mean())
        log.append(vl)
        print(f"  {kind} ep{ep} val_mse {vl:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1])
    return net, log


def predict_net(net, Xn, contact, win):
    N = len(Xn)
    out = np.ones((N, 4), np.float32)
    k, f = np.nonzero((contact > 0.5) & (np.arange(N)[:, None] >= win - 1))
    S = np.column_stack([np.zeros(len(k), int), k, f])
    net.eval()
    with torch.no_grad():
        for i in range(0, len(S), 8192):
            xb = batch([Xn], S[i:i + 8192], win)
            out[S[i:i + 8192, 1], S[i:i + 8192, 2]] = net(xb).numpy()
    return out


def label_metrics(pred, y, idx=None):
    if idx is not None:
        pred, y = pred[idx], y[idx]
    m = ~np.isnan(y)
    p, t = pred[m], y[m]
    return dict(MAE=float(np.mean(np.abs(p - t))), RMSE=float(np.sqrt(np.mean((p - t) ** 2))),
                r=float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 0 else 0.0)


if __name__ == "__main__":
    data = []
    for s in TRAIN:
        X, y = load(s); tr, va = split_idx(len(X)); data.append((X, y, tr, va))
    allX = np.concatenate([X[tr] for X, y, tr, va in data]).reshape(-1, F)
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    mu[FI["leg0"]:] = 0; sd[FI["leg0"]:] = 1
    np.savez(ROOT / "results" / f"feature_norm{TAG}.npz", mu=mu, sd=sd)
    datan = [((X - mu) / sd).astype(np.float32) for X, y, tr, va in data]

    report = {}
    hp = fit_heuristic([(X, y, tr) for X, y, tr, va in data])
    print("heuristic params", hp, flush=True)
    report["heuristic_params"] = hp

    nets = {}
    for kind in ["mlp", "lstm"]:
        net, log = train_net(kind, data, datan, mu, sd)
        torch.save(net.state_dict(), ROOT / "results" / f"{kind}{TAG}.pt")
        nets[kind] = net; report[f"{kind}_val_curve"] = log

    # write scores + label-prediction metrics for every sequence
    for s in SEQS:
        X, y = load(s); Xn = ((X - mu) / sd).astype(np.float32)
        contact = X[..., FI["contact"]]
        y = np.where(contact > 0.5, y, np.nan)
        preds = {"heuristic": heuristic(X, hp).astype(np.float32),
                 "mlp": predict_net(nets["mlp"], Xn, contact, 1),
                 "lstm": predict_net(nets["lstm"], Xn, contact, WIN)}
        for k, v in preds.items():
            np.save(OUT / f"{k}_{s}.npy", v)
        np.save(OUT / f"oracle_{s}.npy", np.nan_to_num(y, nan=1.0).astype(np.float32))
        if s in TRAIN:
            va = split_idx(len(X))[1]
            report[s] = {k: label_metrics(v, y, va) for k, v in preds.items()}
            report[s]["split"] = "validation (last 20%)"
        else:
            report[s] = {k: label_metrics(v, y) for k, v in preds.items()}
            report[s]["split"] = "test"
        print(s, {k: (round(v["MAE"], 3), round(v["r"], 3)) for k, v in report[s].items() if k != "split"}, flush=True)
    json.dump(report, open(ROOT / "results" / f"label_prediction{TAG}.json", "w"), indent=1)
