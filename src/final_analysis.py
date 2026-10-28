"""Phases 10 and 11: statistics, extra analyses and the final verdict for ONE dataset and ONE LLM.
Run through:  python -m src.pipeline analyze --dataset trec [--llm llm2]
Writes into results/<dataset>/runs_<llm>/:
  table1_by_setting.csv    accuracy, macro F1, vote accuracy, label precision, invalid rate
  table_comparisons.csv    per setting: primary (S7 vs S5, S7 vs S3), secondary (S7 vs S2, S7 vs S13) and
                           exploratory comparisons, each family with its own Holm correction
  table_pooled.csv         per comparison, pooled over settings: gap, 95% CI, TOST equivalence, detectable effect
  table_difficulty.csv     IQP minus RBF gap on easy, medium and hard queries
  table_per_class.csv      accuracy per label and selector
  disagreements.csv        questions where S7 and S5 disagree (for error analysis)
  verdict.json             verdicts, checks, predict before you pay correlations, robustness"""
import argparse, glob, json, re, numpy as np, pandas as pd
from scipy.stats import spearmanr
from src.paths import CFG, DATASET, PROFILE, EMB, VARIANT, DATA, RES, RUNS
from src.datasets_registry import spec
from src import metrics_extra as mx
from src.evaluate import accuracy, macro_f1, paired_bootstrap_diff, mcnemar_exact, holm, judge_one_setting, final_verdict

ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=4); ap.add_argument("--method", default="lda")
a = ap.parse_args(); dec = CFG["decision"]; LABELS = spec(DATASET)["labels"]

pat = re.compile(rf"seed\d+_{a.method}\d+_k{a.k}_bw[\d.]+\.csv$")          # main runs only (no pilot, no robust)
files = [f for f in glob.glob(str(RUNS / "*.csv")) if pat.search(f)]
if not files:
    raise SystemExit(f"No main LLM runs found in {RUNS}. Run the main stage first.")
df = pd.concat([pd.read_csv(f) for f in files])
seeds, dims = sorted(df.seed.unique()), sorted(df.dims.unique())
print(f"{DATASET} / {PROFILE}: seeds {seeds}, dims {dims}, {len(df):,} rows")

# ---- Table 1
def summarize(g):
    out = {"acc": accuracy(g.pred, g.gold) * 100, "macro_f1": macro_f1(g.pred, g.gold, LABELS) * 100,
           "vote_acc": g.vote_correct.mean() * 100, "label_prec": g.label_prec.mean() * 100,
           "ret_auroc": g.ret_auroc.mean() * 100,
           "vote_auroc": mx.macro_auroc(mx.str_to_probs(g.vote_p), g.gold.values, LABELS) * 100,
           "invalid_%": (g.pred == "invalid").mean() * 100,
           "llm_auroc": np.nan, "ece": np.nan, "argmax_agrees_with_answer": np.nan}   # same columns in every group
    if "llm_p" in g and g.llm_p.notna().all():          # open LLM: label probabilities exist
        P = mx.str_to_probs(g.llm_p)
        out.update(llm_auroc=mx.macro_auroc(P, g.gold.values, LABELS) * 100, ece=mx.ece(P, g.gold.values, LABELS) * 100,
                   argmax_agrees_with_answer=float(np.mean(np.asarray(LABELS)[P.argmax(1)] == g.pred.values)) * 100)
    return pd.Series(out)
t1 = df.groupby(["dims", "seed", "selector"]).apply(summarize).reset_index()
t1.to_csv(RUNS / "table1_by_setting.csv", index=False)
print(t1.groupby("selector")[[c for c in ["acc", "macro_f1", "ret_auroc", "llm_auroc", "ece", "label_prec"] if c in t1]].mean().round(1).sort_values("acc"))

# ---- Paired comparisons
def compare(a_sel, b_sel):
    rows = []
    for (d, s), g in df.groupby(["dims", "seed"]):
        w = g.pivot(index="query_id", columns="selector", values="correct")
        if a_sel in w and b_sel in w:
            diff, lo, hi = paired_bootstrap_diff(w[a_sel], w[b_sel], seed=int(s))
            rows.append(dict(dims=d, seed=s, a=a_sel, vs=b_sel, diff=diff, lo=lo, hi=hi, p=mcnemar_exact(w[a_sel], w[b_sel])))
    return pd.DataFrame(rows)

def family(name):
    pairs = dec.get(name, [])
    parts = [compare(x, y) for x, y in pairs]
    f = pd.concat([p_ for p_ in parts if len(p_)], ignore_index=True) if any(len(p_) for p_ in parts) else pd.DataFrame()
    if len(f):
        f["p_holm"] = holm(f.p.values); f["family"] = name
    return f

primary, second, explor = family("primary"), family("secondary"), family("exploratory")
comp = pd.concat([f for f in (primary, second, explor) if len(f)], ignore_index=True)
comp["verdict"] = [judge_one_setting(r.diff, r.lo, r.hi, r.p_holm, dec["min_gap_points"], dec["alpha"]) for r in comp.itertuples()]
comp.to_csv(RUNS / "table_comparisons.csv", index=False)
print(comp.round(3).to_string())

# ---- Analyses before the LLM, for the checks and the correlations
pre = {}
for s in seeds:
    for d in dims:
        f = RES / f"pre_llm_seed{s}_{a.method}{d}.json"
        if f.exists():
            pre[(s, d)] = json.load(open(f))

def jac(p, x, y):
    j = p.get("jaccard", {}); return j.get(f"{x}|{y}", j.get(f"{y}|{x}", np.nan))

s7 = t1[t1.selector == "S7_q_iqp"]; s1 = t1[t1.selector == "S1_random"]
checks = {
    "beats random": bool(s7.acc.mean() > s1.acc.mean()),
    "LLM beats vote": bool((s7.acc - s7.vote_acc).mean() > 0),
    "picks differ and g > 1": bool(pre) and all(jac(p, "S5_rbf_lda", "S7_q_iqp") < 0.9 and p["g_quantum_vs_rbf"] > 1 for p in pre.values()),
}
min_seeds = min(dec["min_seeds_agree"], len(seeds)); min_dims = min(dec["min_dims_agree"], len(dims))
verdicts = {}
SHORT = lambda x: x.split("_")[0]
named = []
for fam_name, fam in (("primary", primary), ("secondary", second), ("exploratory", explor)):
    for (x, y), sub in (fam.groupby(["a", "vs"]) if len(fam) else []):
        label = f"{SHORT(x)}_vs_{SHORT(y)}" + ("" if fam_name == "primary" else f" ({fam_name})")
        named.append((label, sub))
for name, sub in named:
    if len(sub):
        v = {(r.seed, r.dims): comp.loc[(comp.a == r.a) & (comp.vs == r.vs) & (comp.seed == r.seed) & (comp.dims == r.dims), "verdict"].iloc[0]
             for r in sub.itertuples()}
        v = {(seeds.index(s), d): x for (s, d), x in v.items()}
        verdicts[name] = final_verdict(v, len(seeds), len(dims), min_seeds, min_dims, checks)
        print(f"FINAL VERDICT {DATASET}/{PROFILE}, {name}: {verdicts[name]}")
print("checks:", checks)

# ---- Predict before you pay: do the free analyses predict the LLM gap?
pbp = {}
rows = []
for r in primary[primary.vs == "S5_rbf_lda"].itertuples():
    p = pre.get((r.seed, r.dims))
    if p and "jaccard" in p:
        rows.append({"gap": r.diff, "abs_gap": abs(r.diff), "jaccard": jac(p, "S5_rbf_lda", "S7_q_iqp"), "g": p["g_quantum_vs_rbf"],
                     "rank": p.get("rank_agreement", {}).get("S7_q_iqp|S5_rbf_lda", np.nan)})
if len(rows) >= 4:
    t = pd.DataFrame(rows)
    pbp = {"n_settings": len(t),
           "spearman_absgap_vs_1minus_jaccard": float(spearmanr(t.abs_gap, 1 - t.jaccard).correlation),
           "spearman_absgap_vs_g": float(spearmanr(t.abs_gap, t.g).correlation),
           "spearman_absgap_vs_1minus_rank": float(spearmanr(t.abs_gap, 1 - t["rank"]).correlation) if t["rank"].notna().all() else None}
    print("predict before you pay:", pbp)

# ---- Pooled over settings: gap, 95% CI, equivalence (TOST) and minimum detectable effect
def per_query_gap(a_sel, b_sel, data=None):
    data = df if data is None else data
    w = data[data.selector.isin([a_sel, b_sel])].pivot_table(index=["seed", "dims", "query_id"], columns="selector", values="correct")
    if a_sel not in w or b_sel not in w:
        return None
    return ((w[a_sel] - w[b_sel]).astype(float).groupby(level="query_id").mean() * 100).values

pooled = []
margin = dec.get("equivalence_margin", dec["min_gap_points"])
for fam_name in ("primary", "secondary", "exploratory"):
    for x, y in dec.get(fam_name, []):
        g_ = per_query_gap(x, y)
        if g_ is None:
            continue
        rng = np.random.default_rng(0); boots = [g_[rng.integers(0, len(g_), len(g_))].mean() for _ in range(2000)]
        p_tost, lo90, hi90 = mx.tost_bootstrap(g_, margin)
        pooled.append({"family": fam_name, "a": x, "vs": y, "gap": g_.mean(), "lo95": np.percentile(boots, 2.5),
                       "hi95": np.percentile(boots, 97.5), "p_tost": p_tost, "lo90": lo90, "hi90": hi90,
                       "equivalent": bool(p_tost < dec["alpha"]), "mde": mx.min_detectable_effect(g_), "n_queries": len(g_)})
pooled = pd.DataFrame(pooled); pooled.to_csv(RUNS / "table_pooled.csv", index=False)
if len(pooled):
    print(pooled.round(3).to_string())

# ---- Difficulty strata: is IQP better on hard queries? Difficulty = how badly a REFERENCE selector outside the
#      comparison (cosine on the full embedding, S2) ranks the pool. Using RBF itself would favour IQP on the "hard"
#      third by regression to the mean, because the strata would be chosen on one arm's own noise.
difficulty = []
if {"S7_q_iqp", "S5_rbf_lda"} <= set(df.selector):
    ref = "S2_cosine_full" if "S2_cosine_full" in set(df.selector) else "S5_rbf_lda"
    d5 = df[df.selector == ref].groupby("query_id").ret_auroc.mean()
    g_ = df[df.selector.isin(["S7_q_iqp", "S5_rbf_lda"])].pivot_table(index=["seed", "dims", "query_id"], columns="selector", values="correct")
    gq = ((g_["S7_q_iqp"] - g_["S5_rbf_lda"]).astype(float).groupby(level="query_id").mean() * 100)
    if d5.notna().sum() > 8:
        q = pd.qcut(d5.rank(method="first"), 3, labels=["hard", "medium", "easy"])
        for lev in ["easy", "medium", "hard"]:
            vals = gq.reindex(q[q == lev].index).dropna().values
            if len(vals):
                rng = np.random.default_rng(1); b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)]
                difficulty.append({"stratum": lev, "n": len(vals), "gap": vals.mean(), "lo": np.percentile(b, 2.5), "hi": np.percentile(b, 97.5),
                                   "ref_ret_auroc": float(d5.reindex(q[q == lev].index).mean()), "reference": ref})
pd.DataFrame(difficulty).to_csv(RUNS / "table_difficulty.csv", index=False)

# ---- Headroom: how much could ANY selector gain? (oracle uses the gold label; it is an upper bound)
headroom = {}
acc_sel = t1.groupby("selector").acc.mean()
if "S14_oracle" in acc_sel:
    real = acc_sel.drop([x for x in ["S14_oracle", "S9_q_iqp_shots"] if x in acc_sel])
    headroom = {"oracle_acc": float(acc_sel["S14_oracle"]), "best_real": real.idxmax(), "best_real_acc": float(real.max()),
                "oracle_minus_best": float(acc_sel["S14_oracle"] - real.max()),
                "oracle_minus_iqp": float(acc_sel["S14_oracle"] - acc_sel.get("S7_q_iqp", np.nan))}
    print("headroom:", headroom)

# ---- Per class accuracy and disagreement examples
pc = df.groupby(["selector", "gold"]).correct.mean().unstack().mul(100).round(1)
pc.to_csv(RUNS / "table_per_class.csv")
test = pd.read_csv(DATA / "test.csv")
last = df[(df.seed == seeds[0]) & (df.dims == dims[-1])]
w = last.pivot(index="query_id", columns="selector", values="correct")
pr = last.pivot(index="query_id", columns="selector", values="pred")
if "S7_q_iqp" in w and "S5_rbf_lda" in w:
    dis = []
    for kind, mask in [("S7 right, S5 wrong", w.S7_q_iqp & ~w.S5_rbf_lda), ("S5 right, S7 wrong", ~w.S7_q_iqp & w.S5_rbf_lda)]:
        for q in w.index[mask]:
            dis.append({"type": kind, "query_id": q, "text": test.text[q], "gold": test.label[q],
                        "pred_S7": pr.S7_q_iqp[q], "pred_S5": pr.S5_rbf_lda[q]})
    pd.DataFrame(dis, columns=["type", "query_id", "text", "gold", "pred_S7", "pred_S5"]).to_csv(RUNS / "disagreements.csv", index=False)
    print(f"disagreements at seed {seeds[0]}, dims {dims[-1]}:",
          pd.DataFrame(dis).type.value_counts().to_dict() if dis else "none")

# ---- Robustness to finite shots (S9), if run
rob = {}
rf = [f for f in glob.glob(str(RUNS / "*_robust.csv"))]
if rf:
    r = pd.concat([pd.read_csv(f) for f in rf])
    wr = r.pivot_table(index=["seed", "dims", "query_id"], columns="selector", values="correct")
    if {"S7_q_iqp", "S9_q_iqp_shots"} <= set(wr.columns):
        diff, lo, hi = paired_bootstrap_diff(wr.S9_q_iqp_shots, wr.S7_q_iqp)
        rob = {"acc_S7_exact": float(wr.S7_q_iqp.mean() * 100), "acc_S9_shots": float(wr.S9_q_iqp_shots.mean() * 100),
               "diff_points": diff, "ci": [lo, hi], "shots": CFG["quantum"]["shots"]}
        keys = {(int(x), int(y)) for x, y in r[["seed", "dims"]].drop_duplicates().values}
        jacs = [jac(p, "S7_q_iqp", "S9_q_iqp_shots") for kk, p in pre.items() if (int(kk[0]), int(kk[1])) in keys]
        rob["jaccard_S7_S9_mean"] = float(np.nanmean(jacs)) if jacs else None
        print("robustness to shots:", rob)

# ---- AUROC comparisons (paired bootstrap over queries, per setting)
auc_rows = []
for (d, s), g in df.groupby(["dims", "seed"]):
    for a_sel, b_sel in [("S7_q_iqp", "S5_rbf_lda"), ("S7_q_iqp", "S3_cosine_lda"), ("S8_q_projected", "S5_rbf_lda")]:
        A, B = g[g.selector == a_sel].sort_values("query_id"), g[g.selector == b_sel].sort_values("query_id")
        if not len(A) or not len(B):
            continue
        r = {"dims": d, "seed": s, "a": a_sel, "vs": b_sel}
        r["d_ret_auroc"], r["d_ret_auroc_lo"], r["d_ret_auroc_hi"] = paired_bootstrap_diff(A.ret_auroc.values, B.ret_auroc.values, seed=int(s))
        if "llm_p" in g and g.llm_p.notna().all() and A.llm_p.notna().all() and B.llm_p.notna().all():
            PA, PB, y = mx.str_to_probs(A.llm_p), mx.str_to_probs(B.llm_p), A.gold.values
            f = lambda idx: (mx.macro_auroc(PA[idx], y[idx], LABELS) - mx.macro_auroc(PB[idx], y[idx], LABELS)) * 100
            r["d_llm_auroc"], r["d_llm_auroc_lo"], r["d_llm_auroc_hi"] = mx.paired_bootstrap_metric(f, len(y), seed=int(s))
        auc_rows.append(r)
pd.DataFrame(auc_rows).to_csv(RUNS / "table_auroc_comparisons.csv", index=False)
auroc_pooled = {}
dfp = df[df.llm_p.notna()] if "llm_p" in df else df.iloc[0:0]
if len(dfp):
    for a_sel, b_sel in [("S7_q_iqp", "S5_rbf_lda"), ("S8_q_projected", "S5_rbf_lda")]:
        blocks = []
        for (d, s), g in dfp.groupby(["dims", "seed"]):
            A, B = g[g.selector == a_sel].sort_values("query_id"), g[g.selector == b_sel].sort_values("query_id")
            if len(A) and len(B):
                blocks.append((mx.str_to_probs(A.llm_p), mx.str_to_probs(B.llm_p), A.gold.values))
        if blocks:
            f = lambda idx: np.mean([(mx.macro_auroc(PA[idx], y[idx], LABELS) - mx.macro_auroc(PB[idx], y[idx], LABELS)) * 100
                                     for PA, PB, y in blocks])
            auroc_pooled[f"{a_sel}|{b_sel}"] = mx.paired_bootstrap_metric(f, len(blocks[0][2]))

# ---- Ordering ablation (if run): accuracy per example order
order = {}
of = glob.glob(str(RUNS / "*_order_*.csv"))
if of:
    o = pd.concat([pd.read_csv(f) for f in of])
    base = df[(df.seed == o.seed.iloc[0]) & (df.dims == o.dims.iloc[0]) & df.selector.isin(o.selector.unique())]
    o = pd.concat([o, base])
    order = (o.groupby(["selector", "order"]).correct.mean() * 100).round(2).unstack().to_dict(orient="index")
    print("ordering ablation (accuracy %):", order)

# ---- Reliability curves for the figures (open LLM only)
calib = {}
if "llm_p" in last and last.llm_p.notna().all():
    for sel in ["S5_rbf_lda", "S7_q_iqp", "S3_cosine_lda"]:
        g = last[last.selector == sel]
        if len(g):
            calib[sel] = mx.reliability(mx.str_to_probs(g.llm_p), g.gold.values, LABELS)

json.dump({"dataset": DATASET, "encoder": EMB, "variant": VARIANT, "llm": PROFILE, "seeds": [int(s) for s in seeds], "dims": [int(d) for d in dims],
           "verdicts": verdicts, "checks": checks, "predict_before_pay": pbp, "robustness_shots": rob,
           "mean_accuracy": t1.groupby("selector").acc.mean().round(2).to_dict(),
           "mean_metrics": t1.groupby("selector").mean(numeric_only=True).drop(columns=["dims", "seed"]).round(3).to_dict(orient="index"),
           "auroc_comparisons": auc_rows, "auroc_pooled": auroc_pooled, "ordering": order, "reliability": calib,
           "pooled": pooled.to_dict(orient="records"), "difficulty": difficulty, "headroom": headroom},
          open(RUNS / "verdict.json", "w"), indent=2, default=float)
print("saved tables in", RUNS)
