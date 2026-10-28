"""Phase 12: figures for ONE dataset and ONE LLM. Run through:  python -m src.pipeline analyze
Saves into figures/<dataset>_<llm>/"""
import glob, json, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.paths import RES, RUNS, FIG, DATASET, PROFILE

NICE = {"S1_random": "Random", "S2_cosine_full": "Cosine (full)", "S3_cosine_lda": "Cosine (LDA)",
        "S4_euclid_lda": "Euclidean", "S5_rbf_lda": "RBF", "S6_q_angle": "Quantum angle", "S7_q_iqp": "Quantum IQP",
        "S8_q_projected": "Quantum projected", "S9_q_iqp_shots": "Quantum IQP (shots)", "S10_bm25": "BM25",
        "S11_mmr_cosine": "MMR (cosine)", "S12_mmr_iqp": "MMR (IQP)", "S13_nca64_full": "NCA-64 metric",
        "S14_oracle": "Label oracle", "S15_rrf_cos_iqp": "Fusion (cos+IQP)",
        "S16_q_trained": "Quantum IQP, trained", "S17_rbf_trained": "RBF, trained"}
def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / f"{name}.png", dpi=200); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)

# Figure 2: accuracy vs number of features (mean over seeds; bars = min and max over seeds)
t1 = pd.read_csv(RUNS / "table1_by_setting.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for sel, g in t1.groupby("selector"):
    m = g.groupby("dims").acc.agg(["mean", "min", "max"])
    ax.errorbar(m.index, m["mean"], yerr=[m["mean"] - m["min"], m["max"] - m["mean"]], marker="o", capsize=3, label=NICE.get(sel, sel))
ax.set_xlabel("Number of features (= qubits)"); ax.set_ylabel("LLM accuracy (%)"); ax.legend(fontsize=7)
ax.set_title(f"{DATASET}, {PROFILE}", fontsize=8); save(fig, "fig2_accuracy_vs_dims")

# Figure 3: overlap heatmap at the largest LDA dimension, seed 0
pre_files = sorted(glob.glob(str(RES / "pre_llm_seed0_lda*.json")), key=lambda p: int(re.search(r"lda(\d+)", p).group(1)))
if pre_files:
    f = pre_files[-1]; J = json.load(open(f)).get("jaccard", {})
    names = sorted({n for k in J for n in k.split("|")})
    if names:
        M = np.eye(len(names))
        for k, v in J.items():
            x, y = k.split("|"); i, j = names.index(x), names.index(y); M[i, j] = M[j, i] = v
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(names))); ax.set_xticklabels([NICE.get(n, n) for n in names], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(names))); ax.set_yticklabels([NICE.get(n, n) for n in names], fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 0.6 else "black")
        fig.colorbar(im, label="Jaccard overlap of picked examples"); ax.set_title(f"{DATASET}: " + f.split("pre_llm_")[1][:-5], fontsize=8)
        save(fig, "fig3_overlap_heatmap")

        # Figure 5: kernel concentration per bandwidth
        rep = json.load(open(f))["bandwidth_report"]; bws = sorted(rep, key=float)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.errorbar([float(b) for b in bws], [rep[b]["mean"] for b in bws], yerr=[rep[b]["std"] for b in bws], marker="o", capsize=3)
        ax.set_xscale("log"); ax.set_xlabel("Bandwidth c"); ax.set_ylabel("IQP kernel value (mean and spread)")
        save(fig, "fig5_kernel_concentration")

# Figure 4: accuracy vs k (seed 0, largest dimension with a k sweep)
kfiles = [f for f in glob.glob(str(RUNS / "seed0_lda*_k*_bw*.csv")) if re.search(r"_bw[\d.]+\.csv$", f)]
if kfiles:
    kd = pd.concat([pd.read_csv(f) for f in kfiles]); kd = kd[kd.k > 0]
    dws = [d for d, g in kd.groupby("dims") if g.k.nunique() > 1]
    if dws:
        g = kd[kd.dims == max(dws)]
        fig, ax = plt.subplots(figsize=(6, 4))
        for sel, gs in g.groupby("selector"):
            m = gs.groupby("k").correct.mean() * 100
            ax.plot(m.index, m.values, marker="o", label=NICE.get(sel, sel))
        ax.set_xlabel("Number of examples in the prompt (k)"); ax.set_ylabel("LLM accuracy (%)")
        ax.set_xticks(sorted(g.k.unique())); ax.legend(fontsize=7); ax.set_title(f"{DATASET}, seed 0, {max(dws)} features", fontsize=8)
        save(fig, "fig4_accuracy_vs_k")

# Figure 6: label precision (no LLM) vs LLM accuracy, one point per selector and setting
fig, ax = plt.subplots(figsize=(5, 4))
for sel, g in t1.groupby("selector"):
    ax.scatter(g.label_prec, g.acc, s=14, label=NICE.get(sel, sel))
ax.set_xlabel("Label precision of picked examples (%)"); ax.set_ylabel("LLM accuracy (%)"); ax.legend(fontsize=6)
save(fig, "fig6_precision_vs_accuracy")
print("figures saved in", FIG)
