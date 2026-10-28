"""Phases 6, 9, 10: analysis, statistics, and the decision rules. Owner: evaluation role."""
import numpy as np
from scipy.stats import binomtest


# ---------- basic metrics ----------
def accuracy(pred, gold):
    return float(np.mean(np.asarray(pred) == np.asarray(gold)))


def macro_f1(pred, gold, labels):
    pred, gold = np.asarray(pred), np.asarray(gold)
    f1s = []
    for l in labels:
        tp = np.sum((pred == l) & (gold == l)); fp = np.sum((pred == l) & (gold != l)); fn = np.sum((pred != l) & (gold == l))
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


# ---------- analyses before the LLM ----------
def jaccard_sets(picked_a, picked_b):
    """Mean Jaccard overlap of the example sets two selectors picked for the same queries."""
    vals = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(picked_a, picked_b)]
    return float(np.mean(vals))


def diversity(picked, y_pool, Z_pool):
    """Distinct labels per picked set, and mean pairwise distance inside each set."""
    distinct = np.mean([len(set(y_pool[i] for i in row)) for row in picked])
    dists = []
    for row in picked:
        Z = Z_pool[row]; D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        dists.append(D[np.triu_indices(len(row), 1)].mean())
    return float(distinct), float(np.mean(dists))


# ---------- statistics ----------
def paired_bootstrap_diff(correct_a, correct_b, n_boot=1000, seed=0):
    """Accuracy of A minus accuracy of B, in POINTS, with a 95% interval.
    Resamples queries, keeping each query's pair of results together."""
    a, b = np.asarray(correct_a, float), np.asarray(correct_b, float)
    rng = np.random.default_rng(seed); n = len(a)
    idx = rng.integers(0, n, (n_boot, n))
    diffs = (a[idx].mean(1) - b[idx].mean(1)) * 100
    return float((a.mean() - b.mean()) * 100), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def mcnemar_exact(correct_a, correct_b):
    """Exact McNemar test: only queries where A and B disagree count."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n10, n01 + n10, 0.5).pvalue)


def holm(pvals):
    """Holm correction for many tests. Returns adjusted p values in the same order."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i])); adj[i] = running
    return adj


# ---------- the decision rules (Phase 10) ----------
def judge_one_setting(diff, lo, hi, p_adj, min_gap, alpha):
    """Verdict for ONE setting (one seed, one dimension)."""
    if lo > 0 and p_adj < alpha and diff >= min_gap:
        return "quantum better"
    if hi < 0 and p_adj < alpha and -diff >= min_gap:
        return "quantum worse"
    if lo > -min_gap and hi < min_gap:
        return "no practical difference"
    return "unclear"


def final_verdict(verdicts_by_setting, n_seeds, n_dims, min_seeds, min_dims, checks):
    """verdicts_by_setting: dict {(seed, dim): verdict}. checks: dict of the 3 extra conditions.
    Returns one of: QUANTUM BETTER, QUANTUM WORSE, NO PRACTICAL DIFFERENCE, MIXED."""
    def enough(v):
        dims_ok = [d for d in {d for _, d in verdicts_by_setting}
                   if sum(verdicts_by_setting[(s, d)] == v for s in range(n_seeds)
                          if (s, d) in verdicts_by_setting) >= min_seeds]
        return len(dims_ok) >= min_dims
    if enough("quantum better") and all(checks.values()):
        return "QUANTUM BETTER"
    if enough("quantum better"):
        return "MIXED (quantum ahead, but a check failed: " + ", ".join(k for k, v in checks.items() if not v) + ")"
    if enough("quantum worse"):
        return "QUANTUM WORSE"
    if enough("no practical difference"):
        return "NO PRACTICAL DIFFERENCE"
    return "MIXED"
