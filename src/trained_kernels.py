"""S16 and S17: the best case for each side. Both kernels get one trainable scale per input feature, trained
to maximise the CENTRED kernel-target alignment (Cortes, Mohri and Rostamizadeh 2012) on the labels of the POOL
only (never the validation or test items):

  S16 (trained quantum):    IQP fidelity kernel on  w * x          (w = d scales, x = standardized features)
  S17 (trained classical):  RBF kernel  exp(-sum_j w_j^2 (x_j - x'_j)^2)   (automatic relevance determination)

Same parameter count, same objective, same optimiser, same data: a fair "trained vs trained" comparison that
answers the objection that a FIXED quantum kernel is a straw man (Hubregtsen et al. 2022 train quantum kernels
the same way). Training uses the NumPy simulator (identical to PennyLane, see tests/test_core.py).
The scales are cached per setting in features/<tag>_trained_<kind>.json."""
import json
import numpy as np
from scipy.optimize import minimize
from src import quantum_kernels as qk


def centred_alignment(K, y):
    """Centred kernel-target alignment, in [-1, 1]; target = +1 same label, -1/(C-1) otherwise, centred."""
    y = np.asarray(y); labs = sorted(set(y)); C = len(labs)
    Y = np.where(y[:, None] == y[None, :], 1.0, -1.0 / max(C - 1, 1))
    n = len(y); H = np.eye(n) - 1.0 / n
    Kc, Yc = H @ K @ H, H @ Y @ H
    den = np.linalg.norm(Kc) * np.linalg.norm(Yc)
    return float((Kc * Yc).sum() / den) if den > 0 else 0.0


def kernel(kind, A, B, w, reps=2, states=None):
    """Kernel between rows of A and B (standardized features) with per-feature scales w."""
    if kind == "rbf":
        D = ((A[:, None, :] - B[None, :, :]) * w) ** 2
        return np.exp(-D.sum(-1))
    st = states or qk.np_states
    return qk.kernel_from_states(st(A * w, "iqp", reps), st(B * w, "iqp", reps))


def fit_scales(kind, S_pool, y_pool, init_scale, seed=0, subsample=250, max_iter=60, reps=2):
    """Per-feature scales maximising centred alignment on a fixed random subsample of the pool."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(S_pool), min(subsample, len(S_pool)), replace=False)
    X, y = S_pool[idx], np.asarray(y_pool)[idx]
    d = X.shape[1]
    theta0 = np.full(d, np.log(init_scale))

    def loss(theta):
        return -centred_alignment(kernel(kind, X, X, np.exp(theta), reps), y)

    res = minimize(loss, theta0, method="L-BFGS-B", bounds=[(np.log(1e-3), np.log(10.0))] * d,
                   options={"maxiter": max_iter, "eps": 1e-3})
    return {"scales": np.exp(res.x).tolist(), "alignment": -float(res.fun), "alignment_init": -float(loss(theta0)),
            "iterations": int(res.nit), "init_scale": float(init_scale), "subsample": int(len(idx))}


def get_scales(kind, feat_dir, tag, S_pool, y_pool, init_scale, seed, cfg, reps=2):
    """Fit once per setting and cache; later calls (other k, other LLMs) reuse the same scales."""
    sub, it = cfg.get("subsample", 250), cfg.get("max_iter", 60)
    f = feat_dir / f"{tag}_trained_{kind}_init{init_scale:.6g}_n{sub}_it{it}_r{reps}.json"   # every setting in the name
    if f.exists():
        return np.array(json.loads(f.read_text())["scales"])
    out = fit_scales(kind, S_pool, y_pool, init_scale, seed, sub, it, reps)
    f.write_text(json.dumps(out, indent=1))
    return np.array(out["scales"])
