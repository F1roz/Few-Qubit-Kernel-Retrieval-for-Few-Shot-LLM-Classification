"""Small helpers shared by paper_tables, paper_text and paper_figures (no argument parsing here)."""
import glob, re
import numpy as np, pandas as pd

DS4 = [("trec", "TREC"), ("emotion", "Emotion"), ("dbpedia", "DBpedia-14"), ("banking77", "Banking77")]


def main_files(folder, method, seed=0, k=4):
    pat = re.compile(rf"seed{seed}_{method}\d+_k{k}_bw[\d.]+\.csv$")
    return [f for f in glob.glob(f"{folder}/*.csv") if pat.search(f)]


def paired_gap(df, a, b, n_boot=1000, seed=0):
    """Mean of a minus b correctness in points, with a bootstrap interval over queries."""
    w = df[df.selector.isin([a, b])].pivot_table(index="query_id", columns="selector", values="correct")
    if a not in w or b not in w:
        return np.nan, np.nan, np.nan
    d = (w[a] - w[b]).astype(float).dropna().values
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return d.mean() * 100, np.percentile(bs, 2.5) * 100, np.percentile(bs, 97.5) * 100


def qubits_llm(ds, llm="llm2"):
    """IQP minus RBF (and trained IQP minus trained RBF) at every qubit count with the LLM, seed 0:
    LDA for d = 2..5, NCA for d = 6..12. Returns None when the NCA runs are missing."""
    folder = f"results/{ds}/runs_{llm}"
    lda, nca = main_files(folder, "lda"), main_files(folder, "nca")
    if not nca:
        return None
    rows = []
    for method, files in (("LDA", lda), ("NCA", nca)):
        for f in files:
            df = pd.read_csv(f); d = int(df.dims.iloc[0])
            g, lo, hi = paired_gap(df, "S7_q_iqp", "S5_rbf_lda")
            gt, lot, hit = paired_gap(df, "S16_q_trained", "S17_rbf_trained")
            acc = df.groupby("selector").correct.mean() * 100
            rows.append({"dims": d, "method": method, "gap": g, "lo": lo, "hi": hi, "gap_trained": gt,
                         "lo_trained": lot, "hi_trained": hit, "acc_iqp": acc.get("S7_q_iqp", np.nan),
                         "acc_rbf": acc.get("S5_rbf_lda", np.nan), "acc_cos_full": acc.get("S2_cosine_full", np.nan)})
    return pd.DataFrame(rows).sort_values("dims").reset_index(drop=True) if rows else None
