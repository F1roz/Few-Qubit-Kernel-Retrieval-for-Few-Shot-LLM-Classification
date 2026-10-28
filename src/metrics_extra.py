"""Metrics beyond accuracy: retrieval AUROC, soft neighbour vote, macro AUROC, calibration (ECE).

Retrieval AUROC (per query): how well a selector's similarity scores rank the pool examples that
share the query's true label above the ones that do not. It needs no LLM and works for every selector,
so it measures the quality of the whole ranking, not only of the top k.
"""
import numpy as np
from scipy.stats import rankdata


def retrieval_auroc(S, y_pool, gold):
    """S: (n_queries, n_pool) scores. Returns one AUROC per query (Mann-Whitney form, ties counted half)."""
    y_pool = np.asarray(y_pool); out = np.full(len(S), np.nan)
    for i, (row, g) in enumerate(zip(S, gold)):
        pos = y_pool == g; n1 = pos.sum(); n0 = len(pos) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(row)
        out[i] = (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def mean_roc_curve(S, y_pool, gold, grid=np.linspace(0, 1, 101)):
    """Vertically averaged retrieval ROC curve over queries: mean true positive rate at each false positive rate."""
    y_pool = np.asarray(y_pool); tprs = []
    for row, g in zip(S, gold):
        pos = y_pool == g
        if pos.all() or not pos.any():
            continue
        order = np.argsort(-row, kind="stable"); p = pos[order]
        tpr = np.r_[0, np.cumsum(p) / p.sum()]; fpr = np.r_[0, np.cumsum(~p) / (~p).sum()]
        tprs.append(np.interp(grid, fpr, tpr))
    return grid, np.mean(tprs, 0)


def soft_vote(picked, y_pool, labels, eps=0.01):
    """Class probabilities from the k picked examples. Rank weights (k for the most similar, 1 for the
    least) make the vote comparable across selectors whose scores live on different scales."""
    k = picked.shape[1]; L = {l: j for j, l in enumerate(labels)}
    P = np.full((len(picked), len(labels)), eps)
    for i, row in enumerate(picked):
        for r, idx in enumerate(row):
            P[i, L[y_pool[idx]]] += k - r
    return P / P.sum(1, keepdims=True)


def macro_auroc(P, gold, labels):
    """One vs rest AUROC averaged over the classes present in gold (Hand and Till style macro average)."""
    gold = np.asarray(gold); aucs = []
    for j, l in enumerate(labels):
        pos = gold == l
        if pos.any() and (~pos).any():
            r = rankdata(P[:, j]); n1, n0 = pos.sum(), (~pos).sum()
            aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs)) if aucs else float("nan")


def ece(P, gold, labels, bins=10):
    """Expected calibration error of the top class probability (Guo et al. 2017)."""
    conf = P.max(1); pred = np.asarray(labels)[P.argmax(1)]; correct = pred == np.asarray(gold)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(P, gold, labels, bins=10):
    conf = P.max(1); correct = np.asarray(labels)[P.argmax(1)] == np.asarray(gold)
    edges = np.linspace(0, 1, bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows


def paired_bootstrap_metric(f, n, n_boot=1000, seed=0):
    """Generic paired bootstrap over queries. f(idx) returns the difference for a resample of query indices.
    Returns (difference on all queries, 2.5 and 97.5 percentiles)."""
    rng = np.random.default_rng(seed)
    full = f(np.arange(n)); boots = [f(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(full), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))


def probs_to_str(P):
    return [" ".join(f"{x:.4f}" for x in row) for row in P]


def str_to_probs(col):
    return np.array([list(map(float, str(s).split())) for s in col])


# ---------------- kernel diagnostics for retrieval (version 5) ----------------
def rank_agreement(SA, SB):
    """Mean over queries of the Spearman correlation between two kernels' rows: how similarly two
    similarity functions ORDER the pool, which is all that retrieval uses."""
    vals = []
    for a, b in zip(SA, SB):
        ra, rb = rankdata(a), rankdata(b)
        if ra.std() > 0 and rb.std() > 0:
            vals.append(np.corrcoef(ra, rb)[0, 1])
    return float(np.mean(vals)) if vals else float("nan")


def kernel_target_alignment(K, y):
    """Kernel-target alignment (Cristianini et al. 2002) with the ideal kernel Y_ij = 1[y_i = y_j]."""
    y = np.asarray(y); Y = (y[:, None] == y[None, :]).astype(float)
    return float((K * Y).sum() / (np.linalg.norm(K) * np.linalg.norm(Y)))


def tost_bootstrap(per_query_gap, margin, n_boot=2000, seed=0):
    """Equivalence test (two one-sided tests) by bootstrap over queries.
    per_query_gap: one value per query (already averaged over settings), in the same unit as margin.
    Returns (p_tost, 90% CI low, 90% CI high). Equivalent at alpha 0.05 when p_tost < 0.05,
    which is the same as the 90% interval lying inside (-margin, +margin)."""
    d = np.asarray(per_query_gap, float); rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    p_low = np.mean(boots <= -margin); p_high = np.mean(boots >= margin)
    return float(max(p_low, p_high)), float(np.percentile(boots, 5)), float(np.percentile(boots, 95))


def min_detectable_effect(per_query_gap, alpha=0.05, power=0.8):
    """Smallest true mean gap a paired test on these queries detects with the given power (normal approx.)."""
    from scipy.stats import norm
    d = np.asarray(per_query_gap, float)
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * d.std(ddof=1) / np.sqrt(len(d)))
