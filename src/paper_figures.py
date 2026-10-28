"""Every figure of the paper, in the IEEE column widths, as vector PDF (and PNG for Word).
Reads the results of all datasets and both LLMs; panels with no results yet say so.
Run:  python -m src.pipeline paper          (writes into paper/figures/)
      python -m src.paper_figures --preview  (stamps PREVIEW on every figure; for layout checks only)"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT = Path("paper/figures"); OUT.mkdir(parents=True, exist_ok=True)
DS = [("trec", "TREC"), ("emotion", "Emotion"), ("dbpedia", "DBpedia-14")]
ONE, TWO = 3.5, 7.16                             # IEEE single and double column widths (inches)
NAME = {"S1_random": "Random", "S2_cosine_full": "Cosine (384-d)", "S3_cosine_lda": "Cosine (LDA)",
        "S4_euclid_lda": "Euclidean", "S5_rbf_lda": "RBF", "S6_q_angle": "Q-Angle", "S7_q_iqp": "Q-IQP",
        "S8_q_projected": "Q-Projected", "S9_q_iqp_shots": "Q-IQP (shots)", "S10_bm25": "BM25",
        "S11_mmr_cosine": "MMR (cosine)", "S12_mmr_iqp": "MMR (Q-IQP)", "S13_nca64_full": "NCA-64",
        "S14_oracle": "Label oracle", "S15_rrf_cos_iqp": "Fusion (cos+IQP)", "S16_q_trained": "Q-IQP trained",
        "S17_rbf_trained": "RBF trained"}
COL = {"S1_random": "#8c8c8c", "S2_cosine_full": "#2a78d6", "S3_cosine_lda": "#1baf7a", "S4_euclid_lda": "#eda100",
       "S5_rbf_lda": "#008300", "S6_q_angle": "#e87ba4", "S7_q_iqp": "#eb6834", "S8_q_projected": "#4a3aa7",
       "S9_q_iqp_shots": "#e34948", "S10_bm25": "#a0522d", "S11_mmr_cosine": "#5fa8e8", "S12_mmr_iqp": "#f4a582",
       "S13_nca64_full": "#17365d", "S14_oracle": "#000000", "S15_rrf_cos_iqp": "#b8860b", "S16_q_trained": "#a3302b",
       "S17_rbf_trained": "#5c9e31"}
MARK = {"S1_random": "", "S2_cosine_full": "D", "S3_cosine_lda": "s", "S4_euclid_lda": "v", "S5_rbf_lda": "o",
        "S6_q_angle": "^", "S7_q_iqp": "P", "S8_q_projected": "X", "S9_q_iqp_shots": "*", "S10_bm25": "<",
        "S11_mmr_cosine": "d", "S12_mmr_iqp": "p", "S13_nca64_full": "h", "S14_oracle": "", "S15_rrf_cos_iqp": ">",
        "S16_q_trained": "P", "S17_rbf_trained": "o"}
DSCOL = {"trec": "#2a78d6", "emotion": "#eb6834", "dbpedia": "#1baf7a", "banking77": "#4a3aa7"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
PREVIEW = False


def save(fig, name):
    if PREVIEW:
        fig.text(0.5, 0.5, "PREVIEW: synthetic data, not results", fontsize=16, color="#d62728", alpha=0.25,
                 ha="center", va="center", rotation=20, weight="bold")
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=300); plt.close(fig)
    print("saved", OUT / f"{name}.pdf")


def empty(ax, msg="not run yet"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="#888888"); ax.set_xticks([]); ax.set_yticks([])


def read(path):
    f = Path(path)
    return pd.read_csv(f) if f.exists() and f.stat().st_size else None


def fig_acc_qubits():
    sels = ["S1_random", "S3_cosine_lda", "S5_rbf_lda", "S6_q_angle", "S7_q_iqp", "S8_q_projected"]
    fig, axes = plt.subplots(1, 3, figsize=(TWO, 1.75), sharey=False)
    for ax, (ds, title) in zip(axes, DS):
        t = read(f"results/{ds}/runs_llm/table1_by_setting.csv")
        ax.set_title(title)
        if t is None:
            empty(ax); continue
        for s in sels:
            g = t[t.selector == s].groupby("dims").acc
            if not len(g.size()):
                continue
            m, lo, hi = g.mean(), g.min(), g.max()
            ax.plot(m.index, m.values, marker=MARK[s] or None, color=COL[s], label=NAME[s],
                    ls="--" if s == "S1_random" else "-", lw=1.0 if s == "S1_random" else 1.4,
                    mec="white", mew=0.5, zorder=3 if s.startswith("S7") else 2)
            if (hi > lo).any():
                ax.fill_between(m.index, lo, hi, color=COL[s], alpha=0.08, lw=0)
        ax.set_xticks(sorted(t.dims.unique())); ax.set_xlabel("Qubits $d$ (LDA dimensions)")
    axes[0].set_ylabel("LLM accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, "fig_acc_qubits")


def fig_roc():
    sels = ["S1_random", "S3_cosine_lda", "S5_rbf_lda", "S6_q_angle", "S7_q_iqp", "S8_q_projected"]
    fig, axes = plt.subplots(1, 3, figsize=(TWO, 1.8))
    for ax, (ds, title) in zip(axes, DS):
        f = Path(f"results/{ds}/figure_material.json"); ax.set_title(title)
        if not f.exists():
            empty(ax); continue
        roc = json.load(open(f)).get("roc", {})
        for s in sels:
            if s in roc:
                r = roc[s]
                ax.plot(r["fpr"], r["tpr"], color=COL[s], ls="--" if s == "S1_random" else "-",
                        lw=1.0 if s == "S1_random" else 1.3, label=f"{NAME[s]} ({r['auroc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#cccccc", lw=0.6, zorder=0)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.set_xlabel("False positive rate"); ax.legend(loc="lower right", fontsize=5.8, handlelength=1.4,
                                                        title="Mean AUROC", title_fontsize=5.8)
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout(); save(fig, "fig_roc")


def fig_ablation():
    frames = [read(f"results/{ds}/ablation_pooled.csv") for ds, _ in DS]
    fig, ax = plt.subplots(figsize=(ONE, 2.9))
    if all(f is None for f in frames):
        empty(ax); save(fig, "fig_ablation"); return
    order = [v for v in frames[next(i for i, f in enumerate(frames) if f is not None)].variant.unique() if not v.startswith("Full")]
    ypos = np.arange(len(order))[::-1]; shapes = ["o", "s", "D"]; cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for j, ((ds, title), f) in enumerate(zip(DS, frames)):
        if f is None:
            continue
        g = f.groupby("variant")[["d_ret_auroc", "d_ret_auroc_lo", "d_ret_auroc_hi"]].mean() * 100
        for y, v in zip(ypos, order):
            if v in g.index:
                r = g.loc[v]; yy = y + (j - 1) * 0.22
                ax.errorbar(r.d_ret_auroc, yy, xerr=[[r.d_ret_auroc - r.d_ret_auroc_lo], [r.d_ret_auroc_hi - r.d_ret_auroc]],
                            fmt=shapes[j], color=cols[j], ms=3.6, mec="white", mew=0.4, elinewidth=0.8, capsize=1.5,
                            label=title if y == ypos[0] else None)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_yticks(ypos); ax.set_yticklabels(order, fontsize=6.4); ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"$\Delta$ retrieval AUROC vs. full method ($\times 100$)")
    ax.legend(loc="lower left", fontsize=6, handletextpad=0.2)
    fig.tight_layout(); save(fig, "fig_ablation")


def fig_concentration_shots():
    f = Path("results/trec/figure_material.json")
    fig = plt.figure(figsize=(ONE, 2.75)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    a, b, c = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    if not f.exists():
        for x in (a, b, c):
            empty(x)
        save(fig, "fig_concentration_shots"); return
    m = json.load(open(f))
    conc = m["concentration"]; names = list(conc)
    parts = a.violinplot([conc[n] for n in names], showmedians=True, widths=0.75)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COL["S5_rbf_lda"] if names[i].startswith("RBF") else COL["S7_q_iqp"]); pc.set_alpha(0.55); pc.set_edgecolor("none")
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#333333"); parts[key].set_linewidth(0.6)
    a.set_xticks(range(1, len(names) + 1))
    a.set_xticklabels([n.replace("IQP c=", "IQP, $c$=") for n in names], fontsize=6.5)
    a.set_ylabel("Off-diagonal $K(x,x')$"); a.set_title("(a) Kernel concentration on the pool (TREC)", fontsize=7.5)
    sh = m["shots"]; N = sorted(map(int, sh["by_shots"]))
    lp = [sh["by_shots"][str(n)]["label_prec"] * 100 for n in N]; sd = [sh["by_shots"][str(n)]["label_prec_sd"] * 100 for n in N]
    jac = [sh["by_shots"][str(n)]["jaccard_vs_exact"] for n in N]
    b.errorbar(N, lp, yerr=sd, color=COL["S9_q_iqp_shots"], marker="o", ms=3, lw=1.1, capsize=1.5, label="$N$ shots")
    b.axhline(sh["exact_label_prec"] * 100, color=COL["S7_q_iqp"], ls="--", lw=0.9, label="exact")
    b.set_ylabel("Precision@$k$ (%)"); b.legend(fontsize=5.8, loc="lower right"); b.set_title("(b) Pick quality", fontsize=7.5)
    c.plot(N, jac, color=COL["S8_q_projected"], marker="o", ms=3, lw=1.1)
    c.set_ylim(0, 1.02); c.set_ylabel("Jaccard with exact picks"); c.set_title("(c) Agreement with exact", fontsize=7.5)
    for x in (b, c):
        x.set_xscale("log", base=2); x.set_xlabel("Shots $N$")
        x.set_xticks([64, 256, 1024, 4096, 16384]); x.set_xticklabels(["64", "256", "1k", "4k", "16k"])
    fig.tight_layout(h_pad=0.8, w_pad=0.8); save(fig, "fig_concentration_shots")


def fig_calibration_precision():
    fig, (a, b) = plt.subplots(1, 2, figsize=(ONE, 1.85))
    f = Path("results/trec/runs_llm2/verdict.json")
    rel = json.load(open(f)).get("reliability", {}) if f.exists() else {}
    if rel:
        a.plot([0, 1], [0, 1], color="#cccccc", lw=0.6)
        for s, pts in rel.items():
            x, y, n = zip(*pts)
            a.plot(x, y, marker=MARK[s], color=COL[s], ms=3.2, lw=1.1, label=NAME[s], mec="white", mew=0.4)
        a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.legend(fontsize=5.6, loc="upper left")
        a.set_xlabel("Confidence"); a.set_ylabel("Accuracy")
    else:
        empty(a)
    a.set_title("(a) Reliability (open LLM, TREC)", fontsize=7.2)
    pts = []
    for ds, _ in DS:
        for llm in ("llm", "llm2"):
            t = read(f"results/{ds}/runs_{llm}/table1_by_setting.csv")
            if t is not None:
                pts.append(t.assign(ds=ds, llm=llm))
    if pts:
        P = pd.concat(pts); shp = {"trec": "o", "emotion": "s", "dbpedia": "D"}
        for (ds, llm), g in P.groupby(["ds", "llm"]):
            b.scatter(g.label_prec, g.acc, s=7, marker=shp[ds], c=[COL.get(s, "#888") for s in g.selector],
                      alpha=0.8 if llm == "llm" else 0.45, linewidths=0)
        rho = spearmanr(P.label_prec, P.acc).correlation
        b.text(0.04, 0.95, rf"Spearman $\rho$ = {rho:.2f}", transform=b.transAxes, va="top", fontsize=6.2)
        from matplotlib.lines import Line2D
        b.legend([Line2D([], [], marker=shp[d], ls="", color="#555", ms=3) for d, _ in DS], [t for _, t in DS],
                 fontsize=5.6, loc="lower right", handletextpad=0.1)
        b.set_xlabel("Label precision@$k$ (%)"); b.set_ylabel("LLM accuracy (%)")
    else:
        empty(b)
    b.set_title("(b) Pick quality vs. accuracy", fontsize=7.2)
    fig.tight_layout(w_pad=0.8); save(fig, "fig_calibration_precision")


def fig_overlap():
    fs = sorted(glob.glob("results/trec/pre_llm_seed0_lda*.json"), key=lambda p: int(re.search(r"lda(\d+)", p).group(1)))
    fig, ax = plt.subplots(figsize=(ONE, 3.0))
    J = json.load(open(fs[-1])).get("jaccard", {}) if fs else {}
    if not J:
        empty(ax); save(fig, "fig_overlap"); return
    names = [n for n in NAME if any(n in k for k in J)]
    M = np.eye(len(names))
    for k, v in J.items():
        x, y = k.split("|"); M[names.index(x), names.index(y)] = M[names.index(y), names.index(x)] = v
    mask = np.triu(np.ones_like(M, bool), 1); Mm = np.ma.array(M, mask=mask)
    im = ax.imshow(Mm, vmin=0, vmax=1, cmap="Blues"); ax.grid(False)
    for i in range(len(names)):
        for j in range(i + 1):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.6, color="white" if M[i, j] > 0.6 else "#222")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([NAME[n] for n in names], rotation=45, ha="right", fontsize=6.2)
    ax.set_yticks(range(len(names))); ax.set_yticklabels([NAME[n] for n in names], fontsize=6.2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Jaccard overlap of picked sets")
    fig.tight_layout(); save(fig, "fig_overlap")


def fig_ksweep():
    """Accuracy against the number of demonstrations k, per dataset (open LLM if it ran the sweep, else the main LLM)."""
    sels = ["S1_random", "S2_cosine_full", "S13_nca64_full", "S5_rbf_lda", "S7_q_iqp", "S11_mmr_cosine", "S14_oracle"]
    fig, axes = plt.subplots(1, 3, figsize=(TWO, 1.85))
    for ax, (ds, title) in zip(axes, DS):
        g = None
        for llm, lname in (("llm2", "open LLM"), ("llm", "main LLM")):
            files = [f for f in glob.glob(f"results/{ds}/runs_{llm}/seed0_lda*_k*_bw*.csv") if re.search(r"_bw[\d.]+\.csv$", f)]
            if not files:
                continue
            kd = pd.concat([pd.read_csv(f) for f in files]); kd = kd[kd.k > 0]
            dws = [d for d, x in kd.groupby("dims") if x.k.nunique() > 1]
            if dws:
                g = kd[kd.dims == max(dws)]; break
        ax.set_title(title if g is None else f"{title} ({lname}, $d$={max(dws)})")
        if g is None:
            empty(ax); continue
        for s_ in sels:
            m = g[g.selector == s_].groupby("k").correct.mean() * 100
            if len(m):
                ax.plot(m.index, m.values, marker=MARK[s_] or None, color=COL[s_], label=NAME[s_],
                        ls=":" if s_ == "S14_oracle" else ("--" if s_ == "S1_random" else "-"), lw=1.0 if s_ in ("S1_random", "S14_oracle") else 1.3,
                        mec="white", mew=0.4, zorder=3 if s_ == "S7_q_iqp" else 2)
        ks = sorted(g.k.unique())
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks); ax.minorticks_off()
        ax.set_xlabel("Demonstrations $k$")
    axes[0].set_ylabel("LLM accuracy (%)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    if h:
        fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "fig_ksweep")


def fig_scaling():
    """Qubit scaling study (no LLM): does the quantum kernel drift away from RBF as the number of qubits grows?"""
    S = {ds: read(f"results/{ds}/scaling.csv") for ds, _ in DS}
    fig, axes = plt.subplots(1, 4, figsize=(TWO, 1.85))
    titles = ["(a) Rank agreement with RBF", "(b) Kernel spread", "(c) Retrieval AUROC", "(d) Kernel-target alignment"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=7.2)
    if all(v is None for v in S.values()):
        for ax in axes:
            empty(ax)
        save(fig, "fig_scaling"); return
    for ds, title in DS:
        s = S[ds]
        if s is None:
            continue
        g = s.groupby("qubits").mean(numeric_only=True); c = DSCOL[ds]
        axes[0].plot(g.index, g.rank_agreement, marker="o", color=c, label=title, mec="white", mew=0.4)
        axes[1].plot(g.index, g.spread_c1, marker="o", color=c, ls="--", lw=1.0, mec="white", mew=0.4)
        axes[1].plot(g.index, g.spread_tuned, marker="o", color=c, mec="white", mew=0.4)
        axes[2].plot(g.index, g.ret_auroc_iqp * 100, marker="P", color=c, mec="white", mew=0.4)
        axes[2].plot(g.index, g.ret_auroc_rbf * 100, marker="o", color=c, ls="--", lw=1.0, mfc="white")
        axes[3].plot(g.index, g.kta_iqp, marker="P", color=c, mec="white", mew=0.4)
        axes[3].plot(g.index, g.kta_rbf, marker="o", color=c, ls="--", lw=1.0, mfc="white")
    axes[0].set_ylabel(r"Mean Spearman $\bar\rho$"); axes[1].set_ylabel("Std of off-diagonal $K$")
    axes[2].set_ylabel(r"AUROC$_\mathrm{ret}$ ($\times$100)"); axes[3].set_ylabel("KTA")
    from matplotlib.lines import Line2D
    axes[1].legend([Line2D([], [], color="#555", ls="-"), Line2D([], [], color="#555", ls="--")], ["tuned $c$", "$c=1$"],
                   fontsize=5.8, loc="upper right")
    axes[2].legend([Line2D([], [], color="#555", ls="-", marker="P"), Line2D([], [], color="#555", ls="--", marker="o", mfc="white")],
                   ["Q-IQP", "RBF"], fontsize=5.8, loc="lower right")
    for ax in axes:
        ax.set_xlabel("Qubits $d$ (NCA features)")
        qs = sorted({int(q) for v in S.values() if v is not None for q in v.qubits.unique()})
        ax.set_xticks(qs)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6); save(fig, "fig_scaling")


def fig_noise():
    """Local depolarizing noise per gate layer: agreement with the noiseless picks, and pick quality."""
    frames = [read(f"results/{ds}/noise.csv") for ds, _ in DS]
    fig, (a, b) = plt.subplots(1, 2, figsize=(ONE, 1.75))
    a.set_title("(a) Agreement with noiseless picks", fontsize=7.0); b.set_title("(b) Pick quality", fontsize=7.0)
    if all(f is None for f in frames):
        empty(a); empty(b); save(fig, "fig_noise"); return
    for (ds, title), n in zip(DS, frames):
        if n is None:
            continue
        c = DSCOL[ds]
        ex = n[n.shots == 0].groupby("p2"); sh = n[n.shots > 0].groupby("p2")
        x = np.array(sorted(n.p2.unique())); xp = np.where(x == 0, x[x > 0].min() / 3 if (x > 0).any() else 1e-4, x)
        a.plot(xp, ex.jaccard_vs_exact.mean().reindex(x), marker="o", color=c, label=f"{title}, exact", mec="white", mew=0.4)
        a.errorbar(xp, sh.jaccard_vs_exact.mean().reindex(x), yerr=sh.jaccard_vs_exact.std().reindex(x).fillna(0),
                   marker="s", color=c, ls="--", lw=1.0, ms=3, capsize=1.2, label=f"{title}, {int(n[n.shots > 0].shots.iloc[0])} shots"
                   if (n.shots > 0).any() else None)
        b.plot(xp, ex.label_prec.mean().reindex(x) * 100, marker="o", color=c, mec="white", mew=0.4)
        b.plot(xp, sh.label_prec.mean().reindex(x) * 100, marker="s", color=c, ls="--", lw=1.0, ms=3)
        ticks = xp; labels = ["0" if v == 0 else f"{v:g}" for v in x]
    for ax in (a, b):
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=6); ax.minorticks_off()
        ax.set_xlabel("Two-qubit error $p_2$")
    a.set_ylim(0, 1.03); a.set_ylabel("Jaccard vs. noiseless"); b.set_ylabel("Precision@$k$ (%)")
    a.legend(fontsize=5.2, loc="lower left", handlelength=1.6)
    fig.tight_layout(w_pad=0.7); save(fig, "fig_noise")


def fig_bandwidth():
    """Bandwidth study with the LLM: IQP accuracy on the test queries at every c, against the RBF line."""
    have = [(ds, t) for ds, t in DS if Path(f"results/{ds}/runs_llm2/bandwidth_study.csv").exists()]
    fig, axes = plt.subplots(1, max(1, len(have)), figsize=(ONE if len(have) <= 1 else TWO * len(have) / 3, 1.8), squeeze=False)
    axes = axes[0]
    if not have:
        empty(axes[0]); axes[0].set_title("Bandwidth study (open LLM)"); save(fig, "fig_bandwidth"); return
    for ax, (ds, title) in zip(axes, have):
        d = read(f"results/{ds}/runs_llm2/bandwidth_study.csv"); j = json.load(open(f"results/{ds}/runs_llm2/bandwidth_study.json"))
        t = d[d.split == "test"]
        for sel in ("S7_q_iqp", "S8_q_projected"):
            g = t[t.selector == sel]
            if len(g):
                m = g.groupby("bandwidth").correct.mean() * 100
                ax.plot(m.index, m.values, marker=MARK[sel], color=COL[sel], label=NAME[sel], mec="white", mew=0.4)
        rbf = t[t.selector == "S5_rbf_lda"].correct.mean() * 100
        ax.axhline(rbf, color=COL["S5_rbf_lda"], ls="--", lw=1.0, label="RBF (tuned $\\gamma$)")
        cv = [v["c_vote"] for v in j["per_dims"].values()]; cl = [v["c_llm"] for v in j["per_dims"].values()]
        import matplotlib.transforms as mtr
        tr = mtr.blended_transform_factory(ax.transData, ax.transAxes)
        ax.scatter(cv, [0.03] * len(cv), marker="^", color="#333", s=14, transform=tr, zorder=4, label="$c$ chosen by vote")
        ax.scatter(cl, [0.10] * len(cl), marker="v", color="#d62728", s=14, transform=tr, zorder=4, label="$c$ chosen by LLM")
        cs = sorted(t.bandwidth.dropna().unique())
        ax.set_xscale("log"); ax.set_xticks(cs); ax.set_xticklabels([f"{c:g}" for c in cs]); ax.minorticks_off()
        ax.set_xlabel("Bandwidth $c$"); ax.set_title(f"{title} (open LLM, mean over $d$)", fontsize=7.0)
    axes[0].set_ylabel("Test accuracy (%)"); axes[0].legend(fontsize=5.4, loc="best")
    fig.tight_layout(); save(fig, "fig_bandwidth")


def fig_difficulty():
    """IQP minus RBF gap within easy, medium and hard queries (terciles of RBF retrieval AUROC)."""
    fig, ax = plt.subplots(figsize=(ONE, 1.7))
    shape = {"trec": "o", "emotion": "s", "dbpedia": "D"}; any_ = False
    strata = ["easy", "medium", "hard"]
    for j, (ds, title) in enumerate(DS):
        d = read(f"results/{ds}/runs_llm/table_difficulty.csv")
        if d is None:
            continue
        any_ = True; d = d.set_index("stratum").reindex(strata)
        x = np.arange(3) + (j - 1) * 0.18
        ax.errorbar(x, d.gap, yerr=[d.gap - d.lo, d.hi - d.gap], fmt=shape[ds], color=DSCOL[ds], ms=3.6, capsize=1.5,
                    elinewidth=0.8, mec="white", mew=0.4, label=title)
    if not any_:
        empty(ax); save(fig, "fig_difficulty"); return
    ax.axhline(0, color="#555", lw=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Easy", "Medium", "Hard"]); ax.grid(axis="x", visible=False)
    ax.set_xlabel("Query difficulty (tercile of RBF retrieval AUROC)"); ax.set_ylabel("IQP $-$ RBF (points)")
    ax.legend(fontsize=5.8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); save(fig, "fig_difficulty")


def fig_baselines():
    """Accuracy of every selector, main LLM, per dataset, with the oracle bound marked."""
    sels = ["S1_random", "S10_bm25", "S3_cosine_lda", "S5_rbf_lda", "S7_q_iqp", "S8_q_projected", "S12_mmr_iqp",
            "S16_q_trained", "S17_rbf_trained", "S2_cosine_full", "S11_mmr_cosine", "S15_rrf_cos_iqp", "S13_nca64_full"]
    fig, ax = plt.subplots(figsize=(TWO, 1.9)); w = 0.8 / len(sels); any_ = False
    for j, s_ in enumerate(sels):
        vals = []
        for ds, _ in DS:
            t = read(f"results/{ds}/runs_llm/table1_by_setting.csv")
            vals.append(t[t.selector == s_].acc.mean() if t is not None and (t.selector == s_).any() else np.nan)
        any_ |= not np.all(np.isnan(vals))
        ax.bar(np.arange(3) + (j - len(sels) / 2 + 0.5) * w, vals, w * 0.88, color=COL[s_], label=NAME[s_],
               hatch="///" if s_.startswith(("S6", "S7", "S8", "S12", "S16")) else None, edgecolor="white", lw=0.3)
    for i, (ds, _) in enumerate(DS):
        t = read(f"results/{ds}/runs_llm/table1_by_setting.csv")
        if t is not None and (t.selector == "S14_oracle").any():
            ax.hlines(t[t.selector == "S14_oracle"].acc.mean(), i - 0.42, i + 0.42, color="black", ls=":", lw=1.0,
                      label="Label oracle" if i == 0 else None)
    if not any_:
        empty(ax)
    ax.set_xticks(range(3)); ax.set_xticklabels([t for _, t in DS]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Accuracy (%)")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.13), fontsize=6)
    fig.tight_layout(rect=(0, 0.12, 1, 1)); save(fig, "fig_baselines")


def fig_datasets():
    sels = ["S1_random", "S2_cosine_full", "S3_cosine_lda", "S5_rbf_lda", "S6_q_angle", "S7_q_iqp", "S8_q_projected"]
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 1.9), sharey=True)
    for ax, (llm, title) in zip(axes, [("llm", "Main LLM"), ("llm2", "Open LLM (Qwen2.5-1.5B)")]):
        ax.set_title(title); any_ = False; w = 0.8 / len(sels)
        for j, s in enumerate(sels):
            vals = []
            for ds, _ in DS:
                t = read(f"results/{ds}/runs_{llm}/table1_by_setting.csv")
                vals.append(t[t.selector == s].acc.mean() if t is not None and (t.selector == s).any() else np.nan)
            if not np.all(np.isnan(vals)):
                any_ = True
            ax.bar(np.arange(3) + (j - len(sels) / 2 + 0.5) * w, vals, w * 0.88, color=COL[s], label=NAME[s])
        ax.set_xticks(range(3)); ax.set_xticklabels([t for _, t in DS]); ax.grid(axis="x", visible=False)
        if not any_:
            empty(ax)
    axes[0].set_ylabel("Accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "fig_datasets")


def fig_qubits_llm():
    """IQP minus RBF with the open LLM from 2 to 12 qubits (LDA up to 5, NCA beyond), per dataset."""
    from src.paper_common import qubits_llm, DS4
    fig, axes = plt.subplots(1, 4, figsize=(TWO, 1.85), sharey=True)
    for ax, (ds, title) in zip(axes, DS4):
        q = qubits_llm(ds); ax.set_title(title)
        if q is None:                                  # shared y axis: do not clear the ticks of the other panels
            ax.text(0.5, 0.5, "not run yet", ha="center", va="center", transform=ax.transAxes, color="#888888")
            ax.set_xticks([]); ax.tick_params(labelleft=False); continue
        ax.axhspan(-2, 2, color="#eeeeee", zorder=0)
        ax.axhline(0, color="#555", lw=0.6)
        ax.errorbar(q.dims, q.gap, yerr=[q.gap - q.lo, q.hi - q.gap], fmt="P-", color=COL["S7_q_iqp"], ms=3.5,
                    lw=1.0, capsize=1.5, elinewidth=0.7, mec="white", mew=0.4, label="IQP $-$ RBF")
        if q.gap_trained.notna().any():
            ax.plot(q.dims, q.gap_trained, "o--", color=COL["S16_q_trained"], ms=3, lw=0.9, mfc="white",
                    label="trained IQP $-$ trained RBF")
        if (q.method == "NCA").any() and (q.method == "LDA").any():
            ax.axvline(5.5, color="#999", ls=":", lw=0.8)
            ax.text(5.75, 0.02, "NCA", transform=ax.get_xaxis_transform(), fontsize=5.5, color="#777", va="bottom")
            ax.text(5.25, 0.02, "LDA", transform=ax.get_xaxis_transform(), fontsize=5.5, color="#777", va="bottom", ha="right")
        ax.set_xticks(sorted(q.dims.unique())); ax.set_xlabel("Qubits $d$")
    axes[0].set_ylabel("Accuracy gap (points)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    if h:
        fig.legend(h, l, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "fig_qubits_llm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--preview", action="store_true")
    PREVIEW = ap.parse_args().preview
    for f in [fig_acc_qubits, fig_roc, fig_ablation, fig_concentration_shots, fig_calibration_precision,
              fig_overlap, fig_ksweep, fig_datasets, fig_scaling, fig_noise, fig_bandwidth, fig_difficulty, fig_baselines,
              fig_qubits_llm]:
        try:
            f()
        except Exception as e:                       # one broken figure must not stop the others
            print(f"FAILED {f.__name__}: {e}")
