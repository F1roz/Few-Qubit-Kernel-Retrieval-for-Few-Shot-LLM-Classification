"""Writes the result-dependent SENTENCES of the paper from the results, with fixed, documented rules.
Output: paper/results_text.tex (macros \\T...) and paper/authors.tex.
The rules only describe what the numbers show (which verdict, which direction, how large); they never
invent a reason. Read every sentence once before submitting: you are the authors.
Run:  python -m src.pipeline paper          (after paper_tables)
      python -m src.paper_text --placeholder (sentences replaced by a red note; the version shipped)"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np, pandas as pd
import yaml

ap = argparse.ArgumentParser(); ap.add_argument("--placeholder", action="store_true")
PH = ap.parse_args().placeholder
CFG = yaml.safe_load(open("configs/config.yaml"))
DS = [("trec", "TREC"), ("emotion", "Emotion"), ("dbpedia", "DBpedia-14")]
NICE = {"S1_random": "random selection", "S2_cosine_full": "cosine on the full embedding",
        "S3_cosine_lda": "cosine on LDA", "S4_euclid_lda": "Euclidean on LDA", "S5_rbf_lda": "RBF",
        "S6_q_angle": "the angle kernel", "S7_q_iqp": "IQP", "S8_q_projected": "the projected kernel"}
VERB = {"QUANTUM BETTER": "quantum better", "QUANTUM WORSE": "quantum worse",
        "NO PRACTICAL DIFFERENCE": "no practical difference"}
LDA_SEL = ["S3_cosine_lda", "S4_euclid_lda", "S5_rbf_lda", "S6_q_angle", "S7_q_iqp", "S8_q_projected"]
T = {}                                                       # macro name -> LaTeX text
MISSING = "\\fillin{written by the code from your results}"
ALWAYS = {"ModelName"}                                       # filled even before any result exists


def tex(s):
    """Escape free text (dataset questions) for LaTeX."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in str(s))


def read(p):
    p = Path(p)
    return pd.read_csv(p) if p.exists() and p.stat().st_size else None


def jload(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def verdict(ds, llm):
    v = jload(f"results/{ds}/runs_{llm}/verdict.json")
    if not v or "S7_vs_S5" not in v.get("verdicts", {}):
        return None
    x = v["verdicts"]["S7_vs_S5"]
    return VERB.get(x, "mixed")


def fmt(x, nd=1, sign=False):
    """Numbers for running text; signed values use a real minus sign."""
    return f"${x:+.{nd}f}$" if sign else f"{x:.{nd}f}"


def big(n):
    return f"{int(n):,}".replace(",", "{,}")


def join(items):
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def rule(name, fn):
    """Run one sentence rule; if its inputs are not there yet, leave a red note."""
    if PH and name not in ALWAYS:
        T[name] = MISSING; return
    try:
        out = fn()
        T[name] = out if out else MISSING
    except Exception as e:                                   # missing results: keep the paper compiling
        print(f"  {name}: not written ({type(e).__name__}: {e})")
        T[name] = MISSING


# ---------------------------------------------------------------- verdict sentences
def other_datasets():
    main = {t: verdict(d, "llm") for d, t in DS}; opn = {t: verdict(d, "llm2") for d, t in DS}
    if main["TREC"] is None:
        return None
    rest = [f"\\emph{{{v}}} on {t}" for t, v in main.items() if t != "TREC" and v]
    s = f"With the main LLM the verdict was {join(rest)}" if rest else ""
    o = [f"\\emph{{{v}}} on {t}" for t, v in opn.items() if v]
    if o:
        s += ("; with the open LLM it was " if s else "With the open LLM the verdict was ") + join(o)
    return s + "." if s else None


def rq1_answer():
    main = {t: verdict(d, "llm") for d, t in DS}
    got = {t: v for t, v in main.items() if v}
    if not got:
        return None
    better = [t for t, v in got.items() if v == "quantum better"]
    worse = [t for t, v in got.items() if v == "quantum worse"]
    none_ = [t for t, v in got.items() if v == "no practical difference"]
    if len(better) >= 2:
        return f"The answer to RQ1 is a qualified yes: the IQP kernel met every preregistered criterion on {join(better)}."
    if better:
        return (f"The IQP kernel met every preregistered criterion on {better[0]} only, so under our rule for general "
                f"claims (two of three datasets) we do not claim an improvement.")
    if len(worse) >= 2:
        return f"The answer to RQ1 is no: the IQP kernel was worse than RBF under our rules on {join(worse)}."
    if len(none_) >= 2:
        return (f"Under our rules the answer to RQ1 is no: the IQP kernel did not improve demonstration selection, "
                f"and the gap was too small to matter on {join(none_)}.")
    parts = [f"\\emph{{{v}}} on {t}" for t, v in got.items()]
    return (f"With verdicts of {join(parts)}, no verdict reached the two datasets our rule requires for a general "
            f"claim, so the preregistered rules alone do not answer RQ1; the pooled equivalence tests below add the missing view.")


# ---------------------------------------------------------------- main table pattern and trend
def main_pattern():
    t = read("results/trec/runs_llm/table1_by_setting.csv")
    m = t.groupby("selector").acc.mean()
    lda = m.reindex(LDA_SEL).dropna()
    spread = lda.max() - lda.min()
    s = f"On TREC the six selectors on LDA features lie within {fmt(spread)} points of each other"
    if "S2_cosine_full" in m:
        d = m["S2_cosine_full"] - lda.max()
        s += (f", and cosine on the full embedding is {fmt(abs(d))} points {'above' if d > 0 else 'below'} "
              f"the best of them ({NICE[lda.idxmax()]})")
    return s + "."


def dims_trend():
    t = read("results/trec/runs_llm/table1_by_setting.csv")
    g = t[t.selector.isin(LDA_SEL)].groupby("dims").acc.mean()
    lo, hi = g.loc[g.index.min()], g.loc[g.index.max()]
    w = t.pivot_table(index="dims", columns="selector", values="acc")
    gap = (w["S7_q_iqp"] - w["S5_rbf_lda"]).abs().max()
    move = "rises" if hi - lo > 1 else ("falls" if lo - hi > 1 else "changes little")
    return (f"on TREC, accuracy averaged over the LDA-based selectors {move} with $d$ (from {fmt(lo)}\\% at "
            f"$d={g.index.min()}$ to {fmt(hi)}\\% at $d={g.index.max()}$), and the IQP and RBF curves are at most "
            f"{fmt(gap)} points apart at any $d$.")


# ---------------------------------------------------------------- diagnostics
def pre_all():
    out = {}
    for d, tname in DS:
        ps = [json.load(open(p)) for p in glob.glob(f"results/{d}/pre_llm_seed*_lda*.json")]
        js = [p["jaccard"].get("S5_rbf_lda|S7_q_iqp", np.nan) for p in ps if "jaccard" in p]
        if ps and js:
            out[tname] = {"J": float(np.nanmean(js)), "g": float(np.mean([p["g_quantum_vs_rbf"] for p in ps]))}
    return out


def diag_interp():
    p = pre_all()
    if not p:
        return None
    Js = [v["J"] for v in p.values()]
    if min(Js) >= 0.8:
        return ("In other words, on these features the IQP kernel and RBF retrieved largely the same demonstrations on "
                "every dataset, which leaves little room for an accuracy difference.")
    if max(Js) < 0.5:
        return ("The IQP kernel therefore retrieved substantially different demonstrations from RBF on every dataset, "
                "so the kernels had room to diverge; Table~\\ref{tab:stats} shows whether that difference reached the LLM.")
    lowest = min(p, key=lambda k: p[k]["J"])
    return (f"The two kernels retrieved partly different demonstrations, most of all on {lowest} "
            f"($J={fmt(p[lowest]['J'], 2)}$).")


def roc_comment():
    t = read("results/trec/runs_llm/table1_by_setting.csv")
    m = t.groupby("selector")[["ret_auroc", "acc"]].mean()
    lda_best = m.reindex(LDA_SEL).ret_auroc.max(); full = m.loc["S2_cosine_full", "ret_auroc"]
    better = lda_best > full
    acc_best_lda = m.reindex(LDA_SEL).acc.max(); acc_full = m.loc["S2_cosine_full", "acc"]
    match = (acc_best_lda > acc_full) == better
    return (f"The LDA-based kernels rank same-label pool items {'better' if better else 'worse'} than cosine on the "
            f"full embedding, which {'matches' if match else 'does not match'} the accuracy ordering in "
            f"Table~\\ref{{tab:main}}" + ("." if match else "; ranking quality over the whole pool is therefore not the "
                                           "only thing that matters for the four examples the LLM sees."))


def pbp():
    v = jload("results/trec/runs_llm/verdict.json"); p = v["predict_before_pay"]
    rj = p["spearman_absgap_vs_1minus_jaccard"]
    if rj is None or np.isnan(rj):
        s = "The gap was the same in every setting, so overlap could not be tested as a screen."
    elif rj >= 0.5:
        s = ("Overlap is therefore a useful screen: the settings in which IQP retrieved different sets were the ones "
             "with larger accuracy gaps.")
    elif rj < 0.3:
        s = "Overlap alone therefore did not predict the size of the gap, so it cannot replace the LLM runs here."
    else:
        s = "The relation between overlap and the gap is positive but weak, so overlap is only a rough screen."
    pts = [read(f"results/{d}/runs_{l}/table1_by_setting.csv") for d, _ in DS for l in ("llm", "llm2")]
    P = pd.concat([x for x in pts if x is not None])
    from scipy.stats import spearmanr
    rp = spearmanr(P.label_prec, P.acc).correlation
    if rp >= 0.7:
        s += " Label precision, which needs no LLM, tracks accuracy closely and ranks selectors almost as the LLM does."
    elif rp < 0.4:
        s += " Label precision tracked accuracy only loosely, so it is no substitute for the LLM runs."
    return s


def screen_ok():
    v = jload("results/trec/runs_llm/verdict.json")
    r = v["predict_before_pay"]["spearman_absgap_vs_1minus_jaccard"]
    return r is not None and not np.isnan(r) and r >= 0.5


# ---------------------------------------------------------------- open LLM
def open_calib():
    v = jload("results/trec/runs_llm2/verdict.json"); est, lo, hi = v["auroc_pooled"]["S7_q_iqp|S5_rbf_lda"]
    t = read("results/trec/runs_llm2/table1_by_setting.csv"); e = t.groupby("selector").ece.mean()
    de = e["S7_q_iqp"] - e["S5_rbf_lda"]
    s = ("The AUROC difference is within noise, so the IQP demonstrations do not change how well the model separates "
         "the classes by a measurable amount" if lo <= 0 <= hi else
         f"The AUROC difference excludes zero, so the IQP demonstrations {'improve' if est > 0 else 'reduce'} how well "
         f"the model separates the classes")
    return s + f"; ECE, which we report but do not test, differs by {fmt(de, 1, True)} points."


# ---------------------------------------------------------------- ablation
ABL_SHORT = {"No entanglement (angle map)": "removing entanglement (angle map)", "IQP, 1 repeat": "one IQP layer",
             "IQP, 3 repeats": "three IQP layers", "Projected kernel instead of fidelity": "the projected kernel",
             "No bandwidth tuning (c = 1)": "dropping bandwidth tuning", "No standardization": "dropping standardization",
             "PCA instead of LDA": "PCA instead of LDA", "First d embedding values": "the first $d$ embedding values instead of LDA",
             "1,024 shots instead of exact": "1{,}024 shots", "RBF on the same features (classical)": "RBF on the same features",
             "RBF on PCA features (classical)": "RBF on PCA features", "Cosine on the same features (classical)": "cosine on the same features"}
REPR = ["No bandwidth tuning (c = 1)", "No standardization", "PCA instead of LDA", "First d embedding values"]
CIRC = ["No entanglement (angle map)", "IQP, 1 repeat", "IQP, 3 repeats"]


def abl_table():
    A = [read(f"results/{d}/ablation_pooled.csv") for d, _ in DS]
    A = pd.concat([a for a in A if a is not None]); A = A[~A.variant.str.startswith("Full")]
    return A.groupby("variant").d_ret_auroc.mean() * 100, read("results/trec/ablation_pooled.csv")


def abl_paragraph():
    mean, trec = abl_table()
    top = mean.abs().sort_values(ascending=False).index[:3]
    parts = [f"{ABL_SHORT[v]} ({fmt(mean[v], 1, True)} points)" for v in top]
    circ = mean.reindex(CIRC).abs().max()
    t = trec.set_index("variant")
    def ci_of(v):
        r = t.loc[v]; return f"[{fmt(r.d_ret_auroc_lo * 100)}, {fmt(r.d_ret_auroc_hi * 100)}]"
    s = (f"Averaged over the three datasets, the largest changes in retrieval AUROC came from {join(parts)}. "
         f"Changing the circuit itself mattered {'far less' if circ < 0.5 * mean.abs().max() else 'comparably'}: "
         f"removing entanglement changed AUROC by {fmt(mean['No entanglement (angle map)'], 1, True)} points "
         f"(TREC interval {ci_of('No entanglement (angle map)')}), and one or three IQP layers by "
         f"{fmt(mean['IQP, 1 repeat'], 1, True)} and {fmt(mean['IQP, 3 repeats'], 1, True)}.")
    return s


def repr_dominates():
    mean, _ = abl_table()
    return mean.reindex(REPR).abs().max() > mean.reindex(CIRC).abs().max()


def abs_ablation():
    mean, _ = abl_table()
    top = mean.abs().sort_values(ascending=False).index[:2]
    s = f"shows that {join(ABL_SHORT[v] for v in top)} change retrieval most"
    s += (", far more than removing entanglement" if repr_dominates() and abs(mean[CIRC[0]]) < 0.5 * mean.abs().max()
          else "")
    return s


def discuss_repr():
    mean, _ = abl_table()
    if repr_dominates():
        return ("The representation decides most of the outcome: supervised compression and correct scaling changed "
                "retrieval more than any change to the circuit.")
    return "Circuit choices mattered at least as much as the representation in our ablation."


# ---------------------------------------------------------------- shots, order, errors
def shots():
    m = jload("results/trec/figure_material.json")["shots"]["by_shots"]
    N = sorted(map(int, m)); ok = [n for n in N if m[str(n)]["jaccard_vs_exact"] >= 0.9]
    if ok:
        n = ok[0]
        return (f"Agreement with the exact picks reached $J\\ge0.9$ at $M={big(n)}$ shots. Each query needs one kernel "
                f"estimate per pool item, so a hardware run at that precision costs about "
                f"{big(500 * n)} shots per query.")
    return (f"Even with $M={big(N[-1])}$ shots the overlap with the exact picks stayed below 0.9 "
            f"({fmt(m[str(N[-1])]['jaccard_vs_exact'], 2)}), so exact retrieval on hardware would need a very large "
            f"shot budget per query.")


def order_cmp():
    v = jload("results/trec/runs_llm2/verdict.json"); o = v["ordering"]
    spread = max(max(r.values()) - min(r.values()) for r in o.values())
    from src.paper_tables import pooled_gap
    fs = [f for f in glob.glob("results/trec/runs_llm2/seed*_lda*_k4_bw*.csv") if re.search(r"_bw[\d.]+\.csv$", f)]
    gap, _, _ = pooled_gap(pd.concat([pd.read_csv(f) for f in fs]), "S7_q_iqp", "S5_rbf_lda")
    if spread > abs(gap):
        return (f"This is larger than the IQP-minus-RBF gap on the same model ({fmt(gap, 1, True)} points), so the "
                f"order of the demonstrations matters more than the kernel that picked them.")
    return (f"This is smaller than the IQP-minus-RBF gap on the same model ({fmt(gap, 1, True)} points), so ordering "
            f"alone cannot explain that gap.")


def errors():
    d = read("results/trec/runs_llm/disagreements.csv")
    if d is None or not len(d):
        return "S7 and S5 gave the same answer on every TREC question at seed 0 and $d=5$."
    top = d.gold.value_counts()
    ex = d.assign(n=d.text.str.len()).sort_values("n").iloc[0]
    n_test = len(pd.read_csv("data/trec/test.csv"))
    cases = "case" if top.iloc[0] == 1 else "cases"
    q = "question" if len(d) == 1 else "questions"
    return (f"At seed 0 and $d=5$, S7 and S5 disagreed on {len(d)} of {n_test} TREC {q}, most often when the gold "
            f"label was \\emph{{{top.index[0]}}} ({top.iloc[0]} {cases}); for example, ``{tex(ex.text)}'' "
            f"(gold: {ex.gold}) was labelled \\emph{{{ex.pred_S7}}} with IQP and \\emph{{{ex.pred_S5}}} with RBF.")


# ---------------------------------------------------------------- discussion and conclusion
def discuss_diag():
    p = pre_all()
    Js = [v["J"] for v in p.values()]; gs = [v["g"] for v in p.values()]
    return (f"across the three datasets the overlap between RBF and IQP ranged from {fmt(min(Js), 2)} to "
            f"{fmt(max(Js), 2)} and the geometric difference from {fmt(min(gs), 2)} to {fmt(max(gs), 2)}.")


def discuss_theory():
    main = [verdict(d, "llm") for d, _ in DS]; main = [v for v in main if v]
    if sum(v == "quantum better" for v in main) >= 2:
        return ("Here the IQP kernel did change what the LLM saw enough to matter, which shows that a fixed quantum "
                "kernel can help retrieval even on compressed classical features; the diagnostics above show where "
                "its geometry departs from RBF.")
    if sum(v == "no practical difference" for v in main) >= 2:
        return ("This agrees with the view that quantum kernels on low-dimensional, well-scaled classical data tend to "
                "behave like smooth classical kernels \\cite{kubler2021inductive,huang2021power,canatar2023bandwidth}. "
                "The setting is in a sense favourable to quantum kernels, since LDA puts class information into every "
                "qubit, and yet the kernel did not change what the LLM saw enough to matter.")
    if sum(v == "quantum worse" for v in main) >= 2:
        return ("Here the IQP kernel was worse than RBF on the same features, so its notion of similarity was a poorer "
                "guide to useful demonstrations; the diagnostics above show where its geometry departs from RBF.")
    return ("The preregistered verdicts do not support a general claim in either direction; the pooled equivalence "
            "tests and the diagnostics above are the better guide to the size and the source of any effect.")


def conclusion_screen():
    return "largely anticipated" if screen_ok() else "only partly anticipated"


def abstract_explain():
    return "anticipate" if screen_ok() else "only partly anticipate"


# ---------------------------------------------------------------- implementation facts
def implementation():
    rows = []
    for line in open(CFG["llm"]["cache_file"]):
        r = json.loads(line)
        if r.get("model") == CFG["llm"]["model"] and r.get("kind") != "probs":
            rows.append(r)
    ks = []
    for f in glob.glob("results/trec/seed*_lda*_k4_*_nollm.csv"):
        x = pd.read_csv(f)
        if "kernel_s" in x:
            ks.append(x.groupby("selector").kernel_s.first().sum())
    s = f"The main LLM answered {big(len(rows))} distinct prompts in total."
    if ks:
        med = np.median(ks)
        s += (f" Simulating all quantum kernels for one setting took {fmt(med, 1 if med < 10 else 0)}~s (median) on "
              f"{CFG['paper']['kernel_machine']}.")
    return s


def model_name():
    p = CFG["paper"]; cache = Path(CFG["llm"]["cache_file"])
    dates = sorted({json.loads(l).get("ts") for l in open(cache) if json.loads(l).get("model") == CFG["llm"]["model"]}
                   - {None}) if cache.exists() and not PH else []
    when = (f", accessed {dates[0]}" + (f" to {dates[-1]}" if dates[-1] != dates[0] else "")) if dates else \
        ", accessed \\fillin{dates are read from the LLM cache}"
    return f"{p['llm_display_name']} (\\texttt{{{tex(CFG['llm']['model'])}}}, {p['llm_provider_name']} API{when})"


# ---------------------------------------------------------------- supplement
def nca():
    t = read("results/trec/runs_llm/table1_by_setting.csv")
    fs = [f for f in glob.glob("results/trec/runs_llm/seed0_nca*_k4_bw*.csv") if re.search(r"_bw[\d.]+\.csv$", f)]
    x = pd.concat([pd.read_csv(f) for f in fs])
    acc = (x.groupby(["dims", "selector"]).correct.mean() * 100).unstack()
    parts = [f"$d={d}$: RBF {fmt(acc.loc[d, 'S5_rbf_lda'])}\\%, IQP {fmt(acc.loc[d, 'S7_q_iqp'])}\\%, projected "
             f"{fmt(acc.loc[d, 'S8_q_projected'])}\\%" for d in acc.index]
    gap_hi = (acc["S7_q_iqp"] - acc["S5_rbf_lda"]).abs().max()
    base = t[t.seed == 0].pivot_table(index="dims", columns="selector", values="acc")
    gap_lo = (base["S7_q_iqp"] - base["S5_rbf_lda"]).abs().max()
    return (f"Accuracies were {'; '.join(parts)}. The largest IQP-minus-RBF gap was {fmt(gap_hi)} points with NCA, "
            f"against {fmt(gap_lo)} points for $d\\le5$ with LDA at the same seed.")


def per_class():
    pc = read("results/trec/runs_llm/table_per_class.csv").set_index("selector")
    keep = [s for s in ["S1_random", "S3_cosine_lda", "S5_rbf_lda", "S7_q_iqp", "S8_q_projected"] if s in pc.index]
    cols = list(pc.columns)
    short = {"abbreviation": "abbr.", "description": "desc.", "location": "loc.", "number": "num."}
    disp = {"S1_random": "Random", "S3_cosine_lda": "Cosine (LDA)", "S5_rbf_lda": "RBF", "S7_q_iqp": "IQP",
            "S8_q_projected": "Projected"}
    head = " & ".join(["Selector"] + [short.get(c, c) for c in cols])
    body = " \\\\\n".join(" & ".join([disp[s]] + [fmt(pc.loc[s, c]) for c in cols]) for s in keep)
    tab = (f"\\begin{{table}}[h]\\centering\\caption{{Per-class accuracy (\\%) on TREC with the main LLM, averaged "
           f"over seeds and $d$.}}\\footnotesize\\setlength{{\\tabcolsep}}{{2.5pt}}\\resizebox{{\\columnwidth}}{{!}}{{"
           f"\\begin{{tabular}}{{@{{}}l{'c' * len(cols)}@{{}}}}\\toprule {head} \\\\\\midrule {body} \\\\\\bottomrule"
           f"\\end{{tabular}}}}\\end{{table}}")
    d = read("results/trec/runs_llm/disagreements.csv")
    ex = ""
    if d is not None and len(d):
        rows = d.assign(n=d.text.str.len()).sort_values("n").head(5)
        ex = ("\n\\noindent Five of the shortest questions on which S7 and S5 disagreed (seed 0, $d=5$):\n\\begin{itemize}\n" +
              "\n".join(f"\\item ``{tex(r.text)}'' (gold: {r.gold}; IQP: {r.pred_S7}; RBF: {r.pred_S5})" for r in rows.itertuples())
              + "\n\\end{itemize}")
    return tab + ex


# ---------------------------------------------------------------- v5: equivalence, baselines, studies
MARGIN = CFG["decision"].get("equivalence_margin", 2.0)


def pooled(ds, llm, a="S7_q_iqp", b="S5_rbf_lda", suffix=""):
    v = jload(f"results/{ds}{suffix}/runs_{llm}/verdict.json") or {}
    return next((p for p in v.get("pooled", []) if p["a"] == a and p["vs"] == b), None)


def pt(x, nd=1):
    """Signed number for running text, with a real minus sign."""
    return f"${x:+.{nd}f}$".replace("$+0.0$", "$0.0$").replace("$-0.0$", "$0.0$")


def cit(lo, hi, nd=1):
    """A confidence interval for running text with real minus signs."""
    f_ = lambda x: (f"$-${abs(x):.{nd}f}" if round(x, nd) < 0 else f"{abs(x):.{nd}f}")
    return f"[{f_(lo)}, {f_(hi)}]"


def rq1_equiv():
    got = {t: pooled(d, "llm") for d, t in DS}; got = {t: p for t, p in got.items() if p}
    if not got:
        return None
    eq = [t for t, p in got.items() if p["equivalent"]]
    up = [t for t, p in got.items() if p["lo95"] > 0 and p["gap"] >= MARGIN]
    small_up = [t for t, p in got.items() if p["lo95"] > 0 and p["gap"] < MARGIN]
    down = [t for t, p in got.items() if p["hi95"] < 0]
    n = len(got)
    if len(up) >= 2:
        return (f"The pooled analysis gives a qualified yes to RQ1: the IQP-minus-RBF gap was above zero and at least "
                f"{MARGIN:g} points on {join(up)}.")
    if len(eq) >= 2:
        s = (f"The pooled analysis answers RQ1 with a positive no: on {join(eq)} the gap was statistically "
             f"equivalent to zero within $\\pm{MARGIN:g}$ points (TOST, $\\alpha=0.05$), so the data rule out an effect "
             f"of the size we set in advance as meaningful, not just fail to find one")
        return s + (f"; on {join(small_up)} a small positive gap below the margin remained." if small_up else ".")
    if len(down) >= 2:
        return f"The pooled analysis answers RQ1 with no: the IQP kernel was below RBF on {join(down)}."
    return (f"The pooled analysis is inconclusive for RQ1: the gap was shown to be negligible on {len(eq)} of {n} "
            f"datasets, and the remaining intervals were too wide to decide.")


def equiv_sentence():
    rows = [(t, pooled(d, "llm")) for d, t in DS]; rows = [(t, p) for t, p in rows if p]
    if not rows:
        return None
    parts = [f"{pt(p['gap'])} points on {t} (95\\% CI {cit(p['lo95'], p['hi95'])}, "
             f"$p_\\mathrm{{TOST}}{'<0.001' if p['p_tost'] < 0.001 else '=' + format(p['p_tost'], '.3f')}$)" for t, p in rows]
    mdes = [p["mde"] for _, p in rows]
    return (f"Pooled over seeds and sizes, the IQP-minus-RBF gap was {join(parts)}. With these sample sizes the study "
            f"had 80\\% power to detect a gap of {fmt(min(mdes))} to {fmt(max(mdes))} points, so an effect at our "
            f"$\\pm{MARGIN:g}$-point margin {'was within reach on every dataset' if max(mdes) <= MARGIN else 'was not within reach on every dataset, and we read the wider intervals accordingly'}.")


def baselines_para():
    out = []
    for d, t in DS:
        v = jload(f"results/{d}/runs_llm/verdict.json")
        if not v:
            continue
        m = v["mean_accuracy"]; h = v["headroom"]
        real = {k: x for k, x in m.items() if k not in ("S14_oracle",)}
        best = max(real, key=real.get); rank = sorted(real, key=real.get, reverse=True).index("S7_q_iqp") + 1
        out.append((t, best, rank, len(real), m.get("S7_q_iqp"), real[best], h["oracle_minus_best"]))
    if not out:
        return None
    BEST = {**NICE, "S10_bm25": "BM25", "S11_mmr_cosine": "MMR on cosine", "S12_mmr_iqp": "MMR on IQP",
            "S13_nca64_full": "the trained NCA-64 metric", "S15_rrf_cos_iqp": "the cosine+IQP fusion",
            "S16_q_trained": "the trained IQP kernel", "S17_rbf_trained": "the trained RBF kernel"}
    s = "; ".join(f"on {t} the best selector was {BEST.get(b, b)} ({fmt(acc)}\\%), IQP ranked {r} of {n} "
                  f"({fmt(q)}\\%), and the label oracle left {fmt(hr)} points of headroom" for t, b, r, n, q, acc, hr in out)
    return s[0].upper() + s[1:] + "."


def secondary_sentence():
    rows = [(t, pooled(d, "llm", "S7_q_iqp", "S2_cosine_full"), pooled(d, "llm", "S7_q_iqp", "S13_nca64_full")) for d, t in DS]
    rows = [r for r in rows if r[1] and r[2]]
    if not rows:
        return None
    below = sum(r[1]["hi95"] < 0 for r in rows) + sum(r[2]["hi95"] < 0 for r in rows)
    parts = [f"{pt(c['gap'])} and {pt(n['gap'])} points on {t}" for t, c, n in rows]
    return (f"Against the strongest classical selectors, cosine on the full embedding and the trained NCA-64 metric, the "
            f"IQP kernel was at {join(parts)}; {below} of {2 * len(rows)} of these intervals lie entirely below zero.")


def fusion_sentence():
    rows = [(t, pooled(d, "llm", "S15_rrf_cos_iqp", "S2_cosine_full"), pooled(d, "llm", "S12_mmr_iqp", "S11_mmr_cosine")) for d, t in DS]
    rows = [r for r in rows if r[1] and r[2]]
    if not rows:
        return None
    up = [t for t, f_, _ in rows if f_["lo95"] > 0]; dn = [t for t, f_, _ in rows if f_["hi95"] < 0]
    s = f"Adding the IQP ranking to cosine by rank fusion changed accuracy by {join(pt(f_['gap']) + ' points on ' + t for t, f_, _ in rows)}"
    if up or dn:
        s += "; the change " + "; ".join(x for x in [f"was above zero on {join(up)}" if up else "",
                                                     f"was below zero on {join(dn)}" if dn else ""] if x) + ". "
    else:
        s += ", and no interval excluded zero. "
    m_up = [t for t, _, m in rows if m["lo95"] > 0]
    s += (f"With diversity added, MMR on IQP minus MMR on cosine was {join(pt(m['gap']) + ' points on ' + t for t, _, m in rows)}"
          + (f", a gain on {join(m_up)}." if m_up else "."))
    return s


def geometry_para():
    ps = {t: [json.load(open(p)) for p in glob.glob(f"results/{d}/pre_llm_seed*_lda*.json")] for d, t in DS}
    ps = {t: v for t, v in ps.items() if v and "rank_agreement" in v[0]}
    if not ps:
        return None
    ra = {t: np.mean([p["rank_agreement"]["S7_q_iqp|S5_rbf_lda"] for p in v]) for t, v in ps.items()}
    rc = {t: np.mean([p["rank_agreement"]["S7_q_iqp|S3_cosine_lda"] for p in v]) for t, v in ps.items()}
    lo = min(ra.values())
    verdict_ = ("at these sizes the tuned IQP kernel orders the pool almost exactly as RBF does" if lo >= 0.9 else
                "the IQP kernel orders the pool largely, but not exactly, as RBF does" if lo >= 0.7 else
                "the IQP kernel orders the pool differently from RBF")
    return (f"On the validation queries, the mean per-query rank correlation between the IQP and RBF scores over the "
            f"pool was {fmt(lo, 3)} to {fmt(max(ra.values()), 3)} across datasets, against "
            f"{fmt(min(rc.values()), 2)} to {fmt(max(rc.values()), 2)} between IQP and cosine on the same features: "
            f"{verdict_}.")


def scaling_para():
    S = {t: read(f"results/{d}/scaling.csv") for d, t in DS}; S = {t: v for t, v in S.items() if v is not None}
    if not S:
        return None
    lo = {t: v.groupby("qubits").rank_agreement.mean() for t, v in S.items()}
    parts = [f"from {fmt(g.iloc[0], 3)} at $d={g.index[0]}$ to {fmt(g.iloc[-1], 3)} at $d={g.index[-1]}$ on {t}" for t, g in lo.items()]
    sp = {t: v.groupby("qubits")[["spread_c1", "spread_tuned"]].mean().iloc[-1] for t, v in S.items()}
    c1 = [x.spread_c1 for x in sp.values()]; tu = [x.spread_tuned for x in sp.values()]
    s = f"As the number of qubits grew, rank agreement with RBF went {join(parts)}. "
    s += (f"At the largest $d$ the off-diagonal kernel values had a spread of {fmt(min(c1), 3)}--{fmt(max(c1), 3)} with "
          f"$c=1$ and {fmt(min(tu), 3)}--{fmt(max(tu), 3)} with the tuned bandwidth"
          + (", so without tuning the kernel concentrates. " if max(c1) < 0.5 * min(tu) else ". "))
    adv = {t: (v.groupby("qubits").ret_auroc_iqp.mean() - v.groupby("qubits").ret_auroc_rbf.mean()) * 100 for t, v in S.items()}
    best_t = max(adv, key=lambda t: adv[t].max()); best = adv[best_t].max(); worst = min(a.min() for a in adv.values())
    drift = max(g.iloc[0] - g.iloc[-1] for g in lo.values())
    if best >= 1.0:
        s += (f"The retrieval AUROC of IQP exceeded RBF by up to {fmt(best)} points ({best_t}, $d={int(adv[best_t].idxmax())}$), "
              f"so at some sizes the quantum ranking was the better one.")
    else:
        s += (f"The retrieval AUROC of IQP minus RBF stayed between {fmt(worst, 1, True)} and {fmt(best, 1, True)} points at "
              f"every size. " + ("The quantum kernel therefore does start to depart from RBF as qubits are added, but "
                                 "the departure did not turn into better retrieval." if drift > 0.02 else
                                 "Adding qubits did not move the quantum kernel away from RBF in any way that matters for retrieval."))
    return s


def noise_para():
    n = read("results/trec/noise.csv"); chk = jload("results/trec/noise_checks.json")
    if n is None or not chk:
        return None
    ex = n[n.shots == 0].groupby("p2").mean(numeric_only=True); sh = n[n.shots > 0].groupby("p2").mean(numeric_only=True)
    ph = ex.index.max(); shots = int(n[n.shots > 0].shots.iloc[0]) if (n.shots > 0).any() else None
    g = min(chk.values())
    s = (f"Global depolarizing noise left every pick unchanged (overlap {fmt(g, 2)}), as Proposition~\\ref{{prop:rank}} requires. "
         if g >= 0.999 else f"Global depolarizing noise changed the picks (overlap {fmt(g, 2)}), which Proposition~"
         f"\\ref{{prop:rank}} rules out; this points to ties or numerical error and should be checked. ")
    J = ex.loc[ph, "jaccard_vs_exact"]
    s += (f"Local noise {'moved' if J < 0.999 else 'did not move'} the picks: at $p_2={ph:g}$ the overlap with the "
          f"noiseless top-$k$ was {fmt(J, 2)} with exact probabilities")
    if shots:
        s += (f" and {fmt(sh.loc[ph, 'jaccard_vs_exact'], 2)} with {big(shots)} swap-test shots (against "
              f"{fmt(sh.loc[sh.index.min(), 'jaccard_vs_exact'], 2)} for shots alone without gate noise)")
    d = (ex.loc[ph, "label_prec"] - ex.loc[ex.index.min(), "label_prec"]) * 100
    s += f", while label precision changed by {pt(d)} points. "
    if shots:
        s += ("Finite sampling, more than gate noise, is the main threat to a hardware version of this selector."
              if sh.loc[sh.index.min(), "jaccard_vs_exact"] < J else
              "Gate noise at this level degrades the picks more than finite sampling does.")
    return s


def bandwidth_para():
    b, at, allc = bw_state()
    n = len(b["per_dims"]); ch = b["n_choices_changed"]
    s = (f"Choosing $c$ by the open LLM's validation accuracy with S7 instead of the neighbour vote changed the choice in "
         f"{ch} of {n} sizes; at the LLM-chosen $c$ the IQP-minus-RBF test gap averaged {pt(b['mean_gap_at_c_llm'])} "
         f"points, and the largest gap at any $c$ was {fmt(b['max_abs_gap_any_c'])} points. ")
    if at and allc:
        return s + "At every bandwidth the interval stayed within our margin, so the result does not depend on how $c$ was tuned."
    if at:
        return s + ("At the LLM-chosen bandwidth the gap stayed within our margin, but at some other bandwidths it did not, "
                    "so a badly chosen $c$ can change the size of the gap.")
    return s + "Tuning the bandwidth with the LLM moved the gap beyond our margin, so the tuning rule matters here."


def robust_para():
    parts = []
    for llm, suf, name in (("llm2", "", "with Qwen2.5-1.5B and three seeds"), ("llm3", "", "with Qwen2.5-7B in 4-bit"),
                           ("llm2", "__mpnet", "with the MPNet encoder (Qwen2.5-1.5B)"),
                           ("llm2", "__big", "with 2{,}000 test items (Qwen2.5-1.5B)")):
        got = [(t, pooled(d, llm, suffix=suf)) for d, t in DS]; got = [(t, p) for t, p in got if p]
        if got:
            parts.append(f"{join(pt(p['gap']) + ' on ' + t for t, p in got)} {name}")
    if not parts:
        return None
    allp = [pooled(d, l, suffix=sf) for d, _ in DS for l, sf in (("llm2", ""), ("llm3", ""), ("llm2", "__mpnet"), ("llm2", "__big"))]
    allp = [p for p in allp if p]
    out = sum(abs(p["gap"]) >= MARGIN and (p["lo95"] > 0 or p["hi95"] < 0) for p in allp)
    return ("The IQP-minus-RBF gap (in points) was " + "; ".join(parts) + ". "
            + ("None of these settings produced a gap that both exceeded the margin and excluded zero."
               if out == 0 else f"{out} of {len(allp)} settings produced a gap that exceeded the margin and excluded zero."))


def ksweep_para():
    fs = []
    for d, t in DS:
        f_ = [x for x in glob.glob(f"results/{d}/runs_llm2/seed0_lda*_k*_bw*.csv") if re.search(r"_bw[\d.]+\.csv$", x)]
        if f_:
            fs.append(pd.concat([pd.read_csv(x) for x in f_]).assign(ds=d))
    if not fs:
        return None
    K = pd.concat(fs); K = K[K.k > 0]; dm = K.groupby("dims").k.nunique().idxmax(); K = K[K.dims == dm]
    w = K[K.selector.isin(["S7_q_iqp", "S5_rbf_lda"])].pivot_table(index=["ds", "k", "query_id"], columns="selector", values="correct")
    g = (w["S7_q_iqp"] - w["S5_rbf_lda"]).groupby(level="k").mean() * 100
    return (f"Across $k\\in\\{{{','.join(str(int(k)) for k in g.index)}\\}}$ the IQP-minus-RBF gap with the open LLM was "
            f"{join(pt(v) + f' at $k={int(k)}$' for k, v in g.items())} points"
            + (", so more demonstrations did not open a gap." if g.abs().max() < MARGIN else
               f"; the gap reached {fmt(g.abs().max())} points, so $k$ matters for the comparison."))


def difficulty_para():
    rows = []
    for d, t in DS:
        x = read(f"results/{d}/runs_llm/table_difficulty.csv")
        if x is not None:
            rows.append((t, x.set_index("stratum")))
    if not rows:
        return None
    hard = [f"{pt(x.loc['hard', 'gap'])} on {t} {cit(x.loc['hard', 'lo'], x.loc['hard', 'hi'])}" for t, x in rows if "hard" in x.index]
    up = [t for t, x in rows if "hard" in x.index and x.loc["hard", "lo"] > 0]
    return (f"On the hardest third of queries, where the reference retriever (cosine on the full embedding) ranked the pool worst, the IQP-minus-RBF gap was {join(hard)}"
            + (f"; the quantum kernel helped on hard queries of {join(up)}." if up else
               "; the quantum kernel did not help on the hardest queries."))


def margin_check():
    t = read("results/trec/runs_llm/table1_by_setting.csv")
    if t is None or t.seed.nunique() < 2:
        raise ValueError("needs at least two TREC seeds")
    n = t.seed.nunique(); sd = t.groupby(["selector", "dims"]).acc.std().mean()
    s = (f"For reference, with the main LLM the accuracy of the same selector varied by {fmt(sd)} points "
         f"(standard deviation over the {({2: 'two', 3: 'three'}).get(n, n)} TREC pools)")
    v = jload("results/trec/runs_llm2/verdict.json") or {}
    o = v.get("ordering", {})
    spread = max(max(r.values()) - min(r.values()) for r in o.values()) if o else None
    if spread is not None:
        s += (f", and with the open LLM, printing the same four demonstrations in another order moved accuracy by "
              f"up to {fmt(spread)} points")
    ref = max(sd, spread or 0)
    return s + (f"; a gap inside $\\pm{MARGIN:g}$ points is thus no larger than variation a practitioner cannot control."
                if ref >= MARGIN else
                f"; the $\\pm{MARGIN:g}$-point margin is therefore stricter than the largest of these sources of variation.")


def qubits_para():
    from src.paper_common import qubits_llm, DS4
    got = {t: qubits_llm(d) for d, t in DS4}; got = {t: q for t, q in got.items() if q is not None}
    if not got:
        return None
    allq = pd.concat([q.assign(ds=t) for t, q in got.items()])
    nca, lda = allq[allq.method == "NCA"], allq[allq.method == "LDA"]
    inside = int(((nca.lo > -MARGIN) & (nca.hi < MARGIN)).sum()); n = len(nca)
    up = nca[(nca.lo > 0) & (nca.gap >= MARGIN)]
    lda_up = lda[(lda.lo > 0) & (lda.gap >= MARGIN)]
    dmin, dmax = int(nca.dims.min()), int(nca.dims.max())
    s = (f"With {dmin} to {dmax} qubits (NCA features) and the open LLM, the IQP-minus-RBF gap ranged from "
         f"{pt(nca.gap.min())} to {pt(nca.gap.max())} points across {join(list(got))}; {inside} of {n} "
         f"interval{'s' if n != 1 else ''} lay entirely within $\\pm{MARGIN:g}$ points. ")
    if len(up):
        r = up.sort_values("gap").iloc[-1]
        s += (f"IQP was ahead by at least {MARGIN:g} points with an interval above zero in {len(up)} "
              f"setting{'s' if len(up) > 1 else ''}, the largest at $d={int(r.dims)}$ on {r.ds} ({pt(r.gap)} points)"
              + (", which the runs with at most five LDA features never showed." if not len(lda_up) else
                 "; such gaps also occurred with LDA features.") )
    elif (nca.lo > 0).any():
        s += "Where an interval lay above zero, the gap stayed below the margin, so more qubits did not open a gap that matters."
    else:
        s += "No interval lay above zero, so more qubits did not open a gap in favour of the quantum kernel."
    if nca.gap_trained.notna().any():
        s += (f" With trained scales on both sides the gap ranged from {pt(nca.gap_trained.min())} to "
              f"{pt(nca.gap_trained.max())} points.")
    return s


def trained_para():
    rows = [(t, pooled(d, "llm", "S16_q_trained", "S17_rbf_trained"), pooled(d, "llm", "S16_q_trained", "S7_q_iqp")) for d, t in DS]
    rows = [r for r in rows if r[1] and r[2]]
    if not rows:
        return None
    def state(p):
        return "ahead" if p["lo95"] > 0 else "behind" if p["hi95"] < 0 else "equivalent" if p["equivalent"] else "open"
    st = {t: state(a) for t, a, _ in rows}
    words = {"ahead": "the trained quantum kernel was ahead", "behind": "the trained quantum kernel was behind",
             "equivalent": "the two were equivalent within the margin", "open": "the data did not decide"}
    groups = [f"on {join([t for t in st if st[t] == k])} {words[k]}" for k in words if any(v == k for v in st.values())]
    s = (f"With the main LLM, trained IQP minus trained RBF was {join(pt(a['gap']) + ' points on ' + t for t, a, _ in rows)}; "
         + "; ".join(groups) + ". ")
    up = [t for t, _, b in rows if b["lo95"] > 0]; dn = [t for t, _, b in rows if b["hi95"] < 0]
    s += (f"Training changed the quantum selector by {join(pt(b['gap']) + ' points on ' + t for t, _, b in rows)} "
          f"relative to the fixed IQP kernel" + (f", a gain that excludes zero on {join(up)}" if up else "")
          + (f", a loss that excludes zero on {join(dn)}" if dn else "") + ".")
    return s


def banking_para():
    t = read("results/banking77/runs_llm2/table1_by_setting.csv"); v = jload("results/banking77/runs_llm2/verdict.json")
    if t is None or not v:
        return None
    m = t.groupby("selector").acc.mean(); real = m.drop(labels=["S14_oracle"], errors="ignore")
    best = real.idxmax()
    BEST = {**NICE, "S10_bm25": "BM25", "S13_nca64_full": "the trained NCA-64 metric", "S16_q_trained": "trained IQP",
            "S17_rbf_trained": "trained RBF", "S2_cosine_full": "cosine on the full embedding",
            "S11_mmr_cosine": "MMR on cosine", "S12_mmr_iqp": "MMR on IQP", "S15_rrf_cos_iqp": "the cosine+IQP fusion"}
    if "S7_q_iqp" not in m or "S5_rbf_lda" not in m:
        return None
    s = f"On Banking77 the open LLM reached {fmt(real[best])}\\% with {BEST.get(best, best)}"
    s += (", and IQP was the best selector" if best == "S7_q_iqp" else f", against {fmt(m['S7_q_iqp'])}\\% for IQP")
    s += f" and {fmt(m['S5_rbf_lda'])}\\% for RBF on at most five LDA features"
    h = v.get("headroom", {}).get("oracle_minus_best")
    s += (f"; the label oracle left {fmt(h)} points of headroom." if h is not None else ".")
    p = next((p for p in v["pooled"] if p["a"] == "S7_q_iqp" and p["vs"] == "S5_rbf_lda"), None)
    if p:
        s += (f" The pooled IQP-minus-RBF gap was {pt(p['gap'])} points (95\\% CI {cit(p['lo95'], p['hi95'])}), "
              + ("above zero." if p["lo95"] > 0 else "below zero." if p["hi95"] < 0 else
                 "equivalent to zero within the margin." if p["equivalent"] else "not decided by the data."))
    b = pooled("banking77", "llm2", suffix="__big")
    if b:
        s += (f" With 2{{,}}000 test items it was {pt(b['gap'])} points (95\\% CI {cit(b['lo95'], b['hi95'])}, "
              f"minimum detectable effect {fmt(b['mde'])} points).")
    return s


def bw_state():
    b = jload("results/trec/runs_llm2/bandwidth_study.json")
    if not b:
        raise ValueError("bandwidth study missing")
    at = abs(b["mean_gap_at_c_llm"]) < MARGIN; allc = b["all_c_within_margin"] is not False
    return b, at, allc


def studies():
    """What the scaling and bandwidth studies show, as facts the three summary sentences use."""
    S = [read(f"results/{d}/scaling.csv") for d, _ in DS]; S = [s for s in S if s is not None]
    b, at, allc = bw_state()
    if not S:
        raise ValueError("scaling study missing")
    A = pd.concat(S)
    g = A.groupby(["dataset", "qubits"])[["rank_agreement", "ret_auroc_iqp", "ret_auroc_rbf"]].mean()
    adv = (g.ret_auroc_iqp - g.ret_auroc_rbf) * 100
    return {"dmax": int(A.qubits.max()), "rank_min": g.rank_agreement.min(), "adv_max": adv.max(),
            "adv_at": adv.idxmax(), "better": adv.max() >= 1.0, "bw_gap": b["mean_gap_at_c_llm"], "bw_ok": at}


def abs_studies():
    s = studies()
    t = (f"Up to {s['dmax']} qubits the tuned IQP kernel kept a mean rank agreement of at least "
         f"{fmt(s['rank_min'], 2)} with RBF")
    t += ((" and its retrieval AUROC never exceeded that of RBF" if s["adv_max"] <= 0.05 else
           f" and its retrieval AUROC exceeded that of RBF by at most {fmt(s['adv_max'])} points") if not s["better"] else
          f", and its retrieval AUROC exceeded RBF by up to {fmt(s['adv_max'])} points ({s['adv_at'][0]}, $d={s['adv_at'][1]}$)")
    t += ("; with the bandwidth chosen by the LLM the gap stayed within the margin." if s["bw_ok"] else
          "; choosing the bandwidth with the LLM moved the gap beyond the margin.")
    return t


def discuss_studies():
    s = studies()
    if s["better"]:
        return (f"The scaling study does show the quantum ranking improving on RBF at some sizes (up to "
                f"{fmt(s['adv_max'])} AUROC points at $d={s['adv_at'][1]}$ on {s['adv_at'][0]}), which marks where an LLM "
                f"study with more qubits would be worth its cost.")
    return (f"Neither study turned the quantum ranking into a better one: up to {s['dmax']} qubits the rank agreement "
            f"stayed at or above {fmt(s['rank_min'], 2)}, and the quantum ranking never retrieved same-label examples "
            f"clearly better than RBF.")


def concl_studies():
    s = studies()
    if s["better"]:
        return (f"Without the LLM, the quantum ranking beat RBF at some sizes (up to {fmt(s['adv_max'])} AUROC points at "
                f"$d={s['adv_at'][1]}$), which is where future LLM studies should look.")
    tail = ("." if s["bw_ok"] else ", although choosing the bandwidth with the LLM moved the accuracy gap beyond our margin.")
    return f"Up to {s['dmax']} qubits and under local noise, the quantum kernel did not become a better retriever than RBF" + tail


def hardware_para():
    h = jload("results/trec/hardware.json")
    if not h:
        return "We did not run the optional hardware check."
    where = "a noiseless simulator of the same circuits" if h["simulated"] else f"the IBM device \\texttt{{{tex(h['backend'])}}}"
    return (f"On {where}, with {h['queries']} queries, {h['pool']} pool items, {h['qubits']} qubits and {big(h['shots'])} shots, "
            f"the top-$k$ overlap with the exact picks was {fmt(h['jaccard_hw_vs_exact'], 2)}, the rank agreement "
            f"{fmt(h['rank_agreement_hw_vs_exact'], 3)} and the mean absolute kernel error {fmt(h['mean_abs_kernel_error'], 3)}.")


VENUES = {
    "ieee": r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage[hidelinks]{hyperref}
\newtheorem{proposition}{Proposition}
\newenvironment{paperproof}{\begin{IEEEproof}}{\end{IEEEproof}}
\newcommand{\PaperBibStyle}{IEEEtran}
\newcommand{\makefront}{\maketitle\begin{abstract}\PaperAbstract\end{abstract}%
\begin{IEEEkeywords}\PaperKeywords\end{IEEEkeywords}}
\newcommand{\makefrontplain}{\maketitle}
\newcommand{\paperack}[1]{\section*{Acknowledgment}#1}
""",
    "springer": r"""\documentclass[runningheads]{llncs}
\usepackage{cite}
\usepackage[hidelinks]{hyperref}
\newenvironment{paperproof}{\begin{proof}}{\end{proof}}
\newcommand{\PaperBibStyle}{splncs04}
\newcommand{\makefront}{\maketitle\begin{abstract}\PaperAbstract\keywords{\PaperKeywords}\end{abstract}}
\newcommand{\makefrontplain}{\maketitle}
\newcommand{\paperack}[1]{\subsubsection*{Acknowledgements}#1}
""",
    "acm": r"""\documentclass[sigconf,nonacm]{acmart}   % remove nonacm for the camera-ready version
\settopmatter{printacmref=false}
\newenvironment{paperproof}{\begin{proof}}{\end{proof}}
\newcommand{\PaperBibStyle}{ACM-Reference-Format}
\newcommand{\makefront}{\begin{abstract}\PaperAbstract\end{abstract}\keywords{\PaperKeywords}\maketitle}
\newcommand{\makefrontplain}{\maketitle}
\newcommand{\paperack}[1]{\begin{acks}#1\end{acks}}
""",
    "arxiv": r"""\documentclass[10pt,twocolumn]{article}
\usepackage[letterpaper,top=0.75in,bottom=1in,left=0.7in,right=0.7in,columnsep=0.25in]{geometry}
\usepackage{times}
\usepackage{cite}
\usepackage{amsthm}
\usepackage[hidelinks]{hyperref}
\newtheorem{proposition}{Proposition}
\newenvironment{paperproof}{\begin{proof}}{\end{proof}}
\newcommand{\PaperBibStyle}{ieeetr}
\newcommand{\makefront}{\maketitle\begin{abstract}\PaperAbstract\par\smallskip\noindent\textbf{Keywords:}
\PaperKeywords\end{abstract}}
\newcommand{\makefrontplain}{\maketitle}
\newcommand{\paperack}[1]{\section*{Acknowledgments}#1}
""",
}


def authors():
    """venue.tex (template) and authors.tex (author block, acknowledgment) from the paper section of the config."""
    p = CFG["paper"]; venue = p.get("venue", "ieee")
    if venue not in VENUES:
        raise SystemExit(f"paper.venue must be one of {sorted(VENUES)}, not {venue!r}")
    Path("paper/venue.tex").write_text(f"% Written by src/paper_text.py: template '{venue}' (configs/config.yaml, paper: venue:).\n"
                                       + VENUES[venue])
    names, mails = p["authors"], p["emails"]
    place = ", ".join(x for x in [p.get("department"), p.get("institution"), p.get("city"), p.get("country")] if x)
    if venue == "ieee":
        block = (f"\\author{{\\IEEEauthorblockN{{{', '.join(names)}}}\n\\IEEEauthorblockA{{{tex(place)}\\\\\n"
                 f"Email: {tex(', '.join(mails))}}}}}")
    elif venue == "springer":
        anded = " \\and ".join(names)
        block = (f"\\author{{{anded}}}\n\\authorrunning{{{names[0]} et al.}}\n"
                 f"\\institute{{{tex(place)}\\\\\n\\email{{{tex(', '.join(mails))}}}}}")
    elif venue == "acm":
        block = "\n".join(f"\\author{{{n}}}\n\\email{{{tex(m)}}}\n\\affiliation{{\\department{{{tex(p.get('department', ''))}}}"
                          f"\\institution{{{tex(p.get('institution', ''))}}}\\city{{{tex(p.get('city', ''))}}}"
                          f"\\country{{{tex(p.get('country', ''))}}}}}" for n, m in zip(names, mails))
    else:
        block = (f"\\author{{{', '.join(names)}\\\\\n{{\\small {tex(place)}}}\\\\\n"
                 f"{{\\small\\texttt{{{tex(', '.join(mails))}}}}}}}")
    ack = " ".join(x for x in [p.get("funding", ""), p.get("ai_disclosure", "")] if x)
    ack += f" Code, cached LLM answers and the preregistration are available at \\RepoURL."
    Path("paper/authors.tex").write_text(
        "% Written by src/paper_text.py from configs/config.yaml (paper section). Edit the config, not this file.\n"
        + block + f"\n\\newcommand{{\\RepoURL}}{{\\url{{{p['repo_url']}}}}}\n\\newcommand{{\\PaperAckText}}{{{ack.strip()}}}\n")
    print(f"wrote paper/venue.tex ({venue}) and paper/authors.tex ({len(names)} authors)")


def write():
    lines = ["% Written by src/paper_text.py from the results. Read each sentence once; do not edit by hand,",
             "% change the results or the rules in src/paper_text.py and rerun  python -m src.pipeline paper",
             "\\providecommand{\\fillin}[1]{{\\color{red}[#1]}}"]
    for k in sorted(T):
        lines.append(f"\\newcommand{{\\T{k}}}{{{T[k]}}}")
    Path("paper/results_text.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(T)} sentences to paper/results_text.tex"
          + (f" ({sum(v == MISSING for v in T.values())} still waiting for results)" if not PH else ""))


if __name__ == "__main__":
    rules = {"OtherDatasets": other_datasets, "RqOne": rq1_answer, "MainPattern": main_pattern, "DimsTrend": dims_trend,
             "DiagInterp": diag_interp, "RocComment": roc_comment, "Pbp": pbp, "OpenCalib": open_calib,
             "AblParagraph": abl_paragraph, "AbsAblation": abs_ablation, "DiscussRepr": discuss_repr, "Shots": shots,
             "Order": order_cmp, "Errors": errors, "DiscussDiag": discuss_diag, "DiscussTheory": discuss_theory,
             "ConclScreen": conclusion_screen, "AbsExplain": abstract_explain,
             "ScreenDid": lambda: "did" if screen_ok() else "did not", "Implementation": implementation,
             "ModelName": model_name, "Nca": nca, "PerClass": per_class, "RqEquiv": rq1_equiv, "Equiv": equiv_sentence,
             "Baselines": baselines_para, "Secondary": secondary_sentence, "Fusion": fusion_sentence,
             "Geometry": geometry_para, "Scaling": scaling_para, "Noise": noise_para, "Bandwidth": bandwidth_para,
             "Robust": robust_para, "KSweep": ksweep_para, "Difficulty": difficulty_para, "Hardware": hardware_para,
             "AbsStudies": abs_studies, "DiscussStudies": discuss_studies, "ConclStudies": concl_studies,
             "MarginCheck": margin_check, "Qubits": qubits_para, "Trained": trained_para, "Banking": banking_para}
    for name, fn in rules.items():
        rule(name, fn)
    write(); authors()
