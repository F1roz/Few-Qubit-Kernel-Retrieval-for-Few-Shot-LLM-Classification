"""Every table of the paper (LaTeX) and every number quoted in its text (results_macros.tex).
Anything not run yet stays as XYZ, so the paper always compiles.
Run:  python -m src.pipeline paper              (writes paper/tables/*.tex and paper/results_macros.tex)
      python -m src.paper_tables --placeholder   (all XYZ, the version shipped with the paper)"""
import argparse, glob, json, re
from pathlib import Path
import numpy as np, pandas as pd, yaml
from scipy.stats import spearmanr

ap = argparse.ArgumentParser(); ap.add_argument("--placeholder", action="store_true")
PH = ap.parse_args().placeholder
OUT = Path("paper/tables"); OUT.mkdir(parents=True, exist_ok=True)
DS = [("trec", "TREC", "Trec"), ("emotion", "Emotion", "Emotion"), ("dbpedia", "DBpedia-14", "Dbpedia")]
LLMS = [("llm", "Main", "main LLM"), ("llm2", "Open", "Qwen2.5-1.5B")]
SEL = [("S1_random", "Random"), ("S2_cosine_full", "Cosine, 384-d"), ("S3_cosine_lda", "Cosine, LDA"),
       ("S4_euclid_lda", "Euclidean, LDA"), ("S5_rbf_lda", "RBF, LDA"), ("S6_q_angle", "Quantum angle"),
       ("S7_q_iqp", "Quantum IQP"), ("S8_q_projected", "Quantum projected")]
X = "XYZ"
CFG_DEC = yaml.safe_load(open("configs/config.yaml")).get("decision", {})
MARGIN = CFG_DEC.get("equivalence_margin", 2.0)
SHORT = {"S1": "Random", "S2": "CosFull", "S3": "CosLda", "S4": "Euclid", "S5": "Rbf", "S6": "Angle", "S7": "Iqp",
         "S8": "Proj", "S9": "IqpShots", "S10": "Bm", "S11": "MmrCos", "S12": "MmrIqp", "S13": "Nca", "S14": "Oracle",
         "S15": "Fusion", "S16": "QTrained", "S17": "RbfTrained"}
M = {}                                                     # macro name -> text


def f(x, nd=1, pm=None):
    if PH or x is None or (isinstance(x, float) and np.isnan(x)):
        return X
    s = f"{x:.{nd}f}"
    return s + (f"$\\pm${pm:.{nd}f}" if pm is not None and not np.isnan(pm) else "")


def num(x, nd=1, plus=False):
    """A number with a real minus sign ($-$), and a plus sign when asked; works in text and in tables."""
    s = f"{abs(x):.{nd}f}"
    if float(s) == 0:
        return s
    return ("$-$" if x < 0 else ("$+$" if plus else "")) + s


def ci(lo, hi, nd=1):
    return X if PH or lo is None or np.isnan(lo) else f"[{num(lo, nd)}, {num(hi, nd)}]"


def sgn(x, nd=1):
    return X if PH or x is None or np.isnan(x) else num(x, nd, plus=True)


def read(p):
    p = Path(p)
    return pd.read_csv(p) if (not PH) and p.exists() and p.stat().st_size else None


def jload(p):
    p = Path(p)
    return json.load(open(p)) if (not PH) and p.exists() else None


def runs(ds, llm):
    fs = [x for x in glob.glob(f"results/{ds}/runs_{llm}/seed*_lda*_k4_bw*.csv") if re.search(r"_bw[\d.]+\.csv$", x)]
    return pd.concat([pd.read_csv(x) for x in fs]) if fs and not PH else None


def pooled_gap(df, a, b, seed=0, n_boot=1000):
    """S_a minus S_b accuracy in points, averaged over every seed and size. The interval comes from a cluster
    bootstrap over test queries: each query's gap is first averaged over all its settings, so the same query
    is never counted as independent evidence twice."""
    w = df[df.selector.isin([a, b])].pivot_table(index=["seed", "dims", "query_id"], columns="selector", values="correct")
    if a not in w or b not in w:
        return np.nan, np.nan, np.nan
    per_query = (w[a] - w[b]).astype(float).groupby(level="query_id").mean().values
    rng = np.random.default_rng(seed)
    boots = [per_query[rng.integers(0, len(per_query), len(per_query))].mean() for _ in range(n_boot)]
    return per_query.mean() * 100, np.percentile(boots, 2.5) * 100, np.percentile(boots, 97.5) * 100


def bold_best(rows, cols):
    """rows: list of lists of strings; bold the largest number in each column index of cols."""
    for c in cols:
        vals = []
        for r in rows:
            m = re.match(r"^(-?[\d.]+)", r[c]); vals.append(float(m.group(1)) if m else -1e9)
        if max(vals) > -1e9:
            best = max(vals)
            for r, v in zip(rows, vals):
                if v == best:
                    r[c] = "\\textbf{" + r[c] + "}"
    return rows


# ---------------- Table: main results (main LLM) ----------------
def tab_main():
    rows = []; t1 = {ds: read(f"results/{ds}/runs_llm/table1_by_setting.csv") for ds, _, _ in DS}
    zs = None if PH else [x for x in glob.glob("results/trec/runs_llm/seed0_lda*_k0_*.csv")]
    z = pd.read_csv(zs[0]) if zs else None
    zrow = ["Zero-shot ($k{=}0$)", f(z.correct.mean() * 100) if z is not None else X, X if z is None else "--", "--"] + ["--"] * 4
    if z is not None:
        from src.evaluate import macro_f1
        from src.datasets_registry import spec
        zrow[2] = f(macro_f1(z.pred, z.gold, spec("trec")["labels"]) * 100)
    M["TrecAccZero"] = zrow[1]
    for sel, name in SEL:
        r = [name]
        for ds, _, D in DS:
            t = t1[ds]
            g = t[t.selector == sel] if t is not None else None
            if g is None or not len(g):
                r += [X, X] + ([X] if ds == "trec" else []); continue
            by_seed = g.groupby("seed")[["acc", "macro_f1", "ret_auroc"]].mean()
            sd = by_seed.std() if len(by_seed) > 1 else pd.Series({"acc": np.nan, "macro_f1": np.nan, "ret_auroc": np.nan})
            r += [f(by_seed.acc.mean(), 1, sd.acc), f(by_seed.macro_f1.mean(), 1, sd.macro_f1)]
            if ds == "trec":
                r += [f(by_seed.ret_auroc.mean(), 1)]
            M[f"{D}Acc{sel.split('_')[0]}"] = f(by_seed.acc.mean())
        rows.append(r)
    rows = bold_best(rows, range(1, 8))
    body = " \\\\\n".join(" & ".join(r) for r in [zrow] + rows)
    tex = r"""\begin{tabular}{@{}l ccc cc cc@{}}
\toprule
 & \multicolumn{3}{c}{TREC (3 seeds)} & \multicolumn{2}{c}{Emotion} & \multicolumn{2}{c}{DBpedia-14} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Selector & Acc & Macro-F1 & AUROC$_\mathrm{ret}$ & Acc & Macro-F1 & Acc & Macro-F1 \\
\midrule
""" + body.replace(" \\\\\nRandom", " \\\\\n\\midrule\nRandom", 1) + r""" \\
\bottomrule
\end{tabular}"""
    tex = tex.replace("Quantum angle", "\\midrule\nQuantum angle", 1)
    (OUT / "tab_main.tex").write_text(tex + "\n")


# ---------------- Table: paired comparisons and verdicts ----------------
def tab_stats():
    rows = []
    for llm, L, Lname in LLMS:
        for ds, title, D in DS:
            df = runs(ds, llm); v = jload(f"results/{ds}/runs_{llm}/verdict.json")
            comp = read(f"results/{ds}/runs_{llm}/table_comparisons.csv")
            cells = [f"{title}", Lname if ds == "trec" else ""]
            for a, b in [("S7_q_iqp", "S5_rbf_lda"), ("S7_q_iqp", "S3_cosine_lda"), ("S8_q_projected", "S5_rbf_lda")]:
                p = next((p for p in (v or {}).get("pooled", []) if p["a"] == a and p["vs"] == b), None)
                if p is not None:                     # the same numbers as the equivalence table (final_analysis)
                    g, lo, hi = p["gap"], p["lo95"], p["hi95"]
                else:
                    g, lo, hi = pooled_gap(df, a, b) if df is not None else (np.nan,) * 3
                cells.append(f"{sgn(g)} {ci(lo, hi)}")
                if a == "S7_q_iqp" and b == "S5_rbf_lda":
                    M[f"Gap{D}{L}"], M[f"CI{D}{L}"] = sgn(g), ci(lo, hi)
            if comp is not None:
                c = comp[(comp.a == "S7_q_iqp") & (comp.vs == "S5_rbf_lda")].verdict.value_counts()
                cells.append(f"{c.get('quantum better', 0)}/{c.get('no practical difference', 0)}/{c.get('quantum worse', 0)}/{c.get('unclear', 0)}")
            else:
                cells.append(X)
            verdict = v["verdicts"].get("S7_vs_S5", X) if v else X
            full = {"QUANTUM BETTER": "quantum better", "QUANTUM WORSE": "quantum worse",
                    "NO PRACTICAL DIFFERENCE": "no practical difference"}.get(verdict, "mixed" if verdict != X else X)
            M[f"Verdict{D}{L}"] = full                       # running text uses the full words
            verdict = {"quantum better": "Q better", "quantum worse": "Q worse", "no practical difference": "No diff.",
                       "mixed": "Mixed"}.get(full, X)
            cells.append(verdict); rows.append(cells)
        rows.append(None)
    lines = []
    for r in rows[:-1]:
        lines.append("\\midrule" if r is None else " & ".join(r) + " \\\\")
    tex = r"""\begin{tabular}{@{}ll ccc cc@{}}
\toprule
Data & LLM & IQP$-$RBF & IQP$-$Cos(LDA) & Proj.$-$RBF & B/E/W/U & Verdict \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    (OUT / "tab_stats.tex").write_text(tex + "\n")


# ---------------- Table: diagnostics before the LLM ----------------
def tab_diag():
    def pre(ds):
        out = []
        for p in sorted(glob.glob(f"results/{ds}/pre_llm_seed*_lda*.json")):
            out.append(json.load(open(p)))
        return out if out and not PH else None

    def nollm(ds):
        fs = glob.glob(f"results/{ds}/seed*_lda*_k4_bw*_nollm.csv")
        return pd.concat([pd.read_csv(x) for x in fs]) if fs and not PH else None

    def jac(p, a, b):
        j = p.get("jaccard", {}); return j.get(f"{a}|{b}", j.get(f"{b}|{a}", np.nan))

    spec_rows = [("$g(K_\\mathrm{RBF}\\,\\|\\,K_\\mathrm{IQP})$, smallest over $\\gamma$", lambda P, N: np.mean([p["g_quantum_vs_rbf"] for p in P]), 2, "g"),
                 ("$g(K_\\mathrm{RBF}\\,\\|\\,K_\\mathrm{IQP})$, median $\\gamma_0$", lambda P, N: np.mean([p["g_quantum_vs_rbf_median"] for p in P]), 2, "gMed"),
                 ("$g(K_\\mathrm{cos}\\,\\|\\,K_\\mathrm{IQP})$", lambda P, N: np.mean([p["g_quantum_vs_cosine"] for p in P]), 2, None),
                 ("$g(K_\\mathrm{RBF}\\,\\|\\,K_\\mathrm{proj})$", lambda P, N: np.mean([p["g_projected_vs_rbf"] for p in P]), 2, None),
                 ("$J$(RBF, angle)", lambda P, N: np.nanmean([jac(p, "S5_rbf_lda", "S6_q_angle") for p in P]), 2, "JRbfAngle"),
                 ("$J$(RBF, IQP)", lambda P, N: np.nanmean([jac(p, "S5_rbf_lda", "S7_q_iqp") for p in P]), 2, "JRbfIqp"),
                 ("$J$(RBF, projected)", lambda P, N: np.nanmean([jac(p, "S5_rbf_lda", "S8_q_projected") for p in P]), 2, None),
                 ("$J$(IQP, IQP 1024 shots)", lambda P, N: np.nanmean([jac(p, "S7_q_iqp", "S9_q_iqp_shots") for p in P]), 2, "JShots"),
                 ("Precision@4, RBF", lambda P, N: N[N.selector == "S5_rbf_lda"].label_prec.mean() * 100, 1, None),
                 ("Precision@4, IQP", lambda P, N: N[N.selector == "S7_q_iqp"].label_prec.mean() * 100, 1, None),
                 ("Precision@4, projected", lambda P, N: N[N.selector == "S8_q_projected"].label_prec.mean() * 100, 1, None),
                 ("AUROC$_\\mathrm{ret}$, RBF", lambda P, N: N[N.selector == "S5_rbf_lda"].ret_auroc.mean() * 100, 1, "RetRbf"),
                 ("AUROC$_\\mathrm{ret}$, IQP", lambda P, N: N[N.selector == "S7_q_iqp"].ret_auroc.mean() * 100, 1, "RetIqp"),
                 ("AUROC$_\\mathrm{ret}$, cosine 384-d", lambda P, N: N[N.selector == "S2_cosine_full"].ret_auroc.mean() * 100, 1, "RetCosFull")]
    data = {ds: (pre(ds), nollm(ds)) for ds, _, _ in DS}
    lines = []; vals_by_macro = {}
    for name, fn, nd, mac in spec_rows:
        cells = [name]
        for ds, _, D in DS:
            P, N = data[ds]
            try:
                v = fn(P, N) if P is not None and N is not None else np.nan
            except Exception:
                v = np.nan
            cells.append(f(v, nd))
            if mac:
                M[f"{mac}{D}"] = f(v, nd); vals_by_macro.setdefault(mac, []).append(v)
        lines.append(" & ".join(cells) + " \\\\")
        if name.startswith("$J$(IQP") or name.startswith("$g(K_\\mathrm{proj}"):
            lines.append("\\midrule")
    for mac, vs in vals_by_macro.items():
        vs = [v for v in vs if not np.isnan(v)]
        M[f"{mac}Min"], M[f"{mac}Max"] = (f(min(vs), 2), f(max(vs), 2)) if vs else (X, X)
    tex = r"""\begin{tabular}{@{}l ccc@{}}
\toprule
Diagnostic (no LLM) & TREC & Emotion & DBpedia-14 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    (OUT / "tab_diag.tex").write_text(tex + "\n")


# ---------------- Table: ablation ----------------
ABL_MACRO = {"No entanglement (angle map)": "AblAngle", "IQP, 1 repeat": "AblRepOne", "IQP, 3 repeats": "AblRepThree",
             "Projected kernel instead of fidelity": "AblProj", "No bandwidth tuning (c = 1)": "AblNoTune",
             "No standardization": "AblNoStd", "PCA instead of LDA": "AblPca", "First d embedding values": "AblTrunc",
             "1,024 shots instead of exact": "AblShots", "RBF on the same features (classical)": "AblRbf",
             "RBF on PCA features (classical)": "AblRbfPca", "Cosine on the same features (classical)": "AblCos"}


def tab_ablation():
    A = {ds: read(f"results/{ds}/ablation.csv") for ds, _, _ in DS}
    AP = {ds: read(f"results/{ds}/ablation_pooled.csv") for ds, _, _ in DS}
    lines = []; group_prev = None
    names = ["Full method (IQP, 2 reps, tuned c, LDA)"] + list(ABL_MACRO)
    groups = {"Full method (IQP, 2 reps, tuned c, LDA)": "reference", "No entanglement (angle map)": "map", "IQP, 1 repeat": "map",
              "IQP, 3 repeats": "map", "Projected kernel instead of fidelity": "kernel", "No bandwidth tuning (c = 1)": "scaling",
              "No standardization": "scaling", "PCA instead of LDA": "features", "First d embedding values": "features",
              "1,024 shots instead of exact": "hardware", "RBF on the same features (classical)": "classical",
              "RBF on PCA features (classical)": "classical", "Cosine on the same features (classical)": "classical"}
    for n in names:
        g = groups[n]
        if group_prev is not None and g != group_prev:
            lines.append("\\midrule")
        group_prev = g
        cells = [n.replace("tuned c", "tuned $c$").replace("c = 1", "$c{=}1$").replace("First d", "First $d$").replace("1,024", "1{,}024")]
        t = A["trec"]
        r = t[t.variant == n].mean(numeric_only=True) if t is not None and (t.variant == n).any() else None
        if r is None:
            cells += [X] * 4
        else:
            p = AP["trec"]; q = p[p.variant == n].mean(numeric_only=True) if p is not None and (p.variant == n).any() else None
            cells += [f(r.ret_auroc * 100), f(r.label_prec * 100), f(r.jaccard_vs_full, 2),
                      "--" if n.startswith("Full") else (X if q is None else f"{sgn(q.d_ret_auroc * 100)} {ci(q.d_ret_auroc_lo * 100, q.d_ret_auroc_hi * 100)}")]
            if n in ABL_MACRO:
                M[ABL_MACRO[n]] = sgn(r.d_ret_auroc * 100)
        for ds in ("emotion", "dbpedia"):
            t = A[ds]
            r = t[t.variant == n].mean(numeric_only=True) if t is not None and (t.variant == n).any() else None
            cells.append(X if r is None else ("--" if n.startswith("Full") else sgn(r.d_ret_auroc * 100)))
        lines.append(" & ".join(cells) + " \\\\")
    tex = r"""\begin{tabular}{@{}l cccc cc@{}}
\toprule
 & \multicolumn{4}{c}{TREC} & Emotion & DBpedia-14 \\
\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}
Variant & AUROC$_\mathrm{ret}$ & Prec@4 & $J$ vs.\ full & $\Delta$AUROC [95\% CI] & $\Delta$AUROC & $\Delta$AUROC \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    (OUT / "tab_ablation.tex").write_text(tex + "\n")


# ---------------- Table: open LLM, probabilistic metrics ----------------
def tab_open():
    sels = [s for s in SEL if s[0] in ("S1_random", "S2_cosine_full", "S3_cosine_lda", "S5_rbf_lda", "S6_q_angle", "S7_q_iqp", "S8_q_projected")]
    T = {ds: read(f"results/{ds}/runs_llm2/table1_by_setting.csv") for ds, _, _ in DS}
    rows = []
    for sel, name in sels:
        r = [name]
        for ds, _, D in DS:
            t = T[ds]; g = t[t.selector == sel] if t is not None else None
            if g is None or not len(g) or "llm_auroc" not in g:
                r += [X, X]; continue
            r += [f(g.llm_auroc.mean()), f(g.ece.mean())]
            M[f"Open{D}Acc{sel.split('_')[0]}"] = f(g.acc.mean())
            M[f"Open{D}Auroc{sel.split('_')[0]}"], M[f"Open{D}Ece{sel.split('_')[0]}"] = f(g.llm_auroc.mean()), f(g.ece.mean())
        rows.append(r)
    rows = bold_best(rows, [1, 3, 5])
    tex = r"""\begin{tabular}{@{}l cc cc cc@{}}
\toprule
 & \multicolumn{2}{c}{TREC} & \multicolumn{2}{c}{Emotion} & \multicolumn{2}{c}{DBpedia-14} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Selector & AUROC & ECE$\downarrow$ & AUROC & ECE$\downarrow$ & AUROC & ECE$\downarrow$ \\
\midrule
""" + " \\\\\n".join(" & ".join(r) for r in rows) + r""" \\
\bottomrule
\end{tabular}"""
    (OUT / "tab_open.tex").write_text(tex + "\n")
    acc_rows = []
    for sel, name in sels:
        r = [name]
        for ds, _, D in DS:
            t = T[ds]; g = t[t.selector == sel] if t is not None else None
            r += [X, X] if g is None or not len(g) else [f(g.acc.mean()), f(g.macro_f1.mean())]
        acc_rows.append(r)
    acc_rows = bold_best(acc_rows, range(1, 7))
    (OUT / "tab_open_acc.tex").write_text(r"""\begin{tabular}{@{}l cc cc cc@{}}
\toprule
 & \multicolumn{2}{c}{TREC} & \multicolumn{2}{c}{Emotion} & \multicolumn{2}{c}{DBpedia-14} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Selector & Acc & Macro-F1 & Acc & Macro-F1 & Acc & Macro-F1 \\
\midrule
""" + " \\\\\n".join(" & ".join(r) for r in acc_rows) + r""" \\
\bottomrule
\end{tabular}""" + "\n")
    # AUROC gap of IQP vs RBF for the open LLM on TREC
    v = jload("results/trec/runs_llm2/verdict.json") or {}
    pa = v.get("auroc_pooled", {}).get("S7_q_iqp|S5_rbf_lda")
    if pa:
        M["OpenAurocGapTrec"], M["OpenAurocGapCITrec"] = sgn(pa[0]), ci(pa[1], pa[2])


# ---------------- Table: example order (open LLM) ----------------
def tab_order():
    v = jload("results/trec/runs_llm2/verdict.json"); o = (v or {}).get("ordering", {})
    names = dict(SEL); rows = []
    spreads = []
    for sel in ["S3_cosine_lda", "S5_rbf_lda", "S7_q_iqp"]:
        r = o.get(sel, {})
        vals = [r.get(k, np.nan) for k in ("near_last", "near_first", "shuffled")]
        rows.append([names[sel]] + [f(x) for x in vals])
        if not np.all(np.isnan(vals)):
            spreads.append(np.nanmax(vals) - np.nanmin(vals))
    M["OrderSpreadMax"] = f(max(spreads)) if spreads else X
    tex = r"""\begin{tabular}{@{}l ccc@{}}
\toprule
Selector & Nearest last & Nearest first & Shuffled \\
\midrule
""" + " \\\\\n".join(" & ".join(r) for r in rows) + r""" \\
\bottomrule
\end{tabular}"""
    (OUT / "tab_order.tex").write_text(tex + "\n")


# ---------------- Table: stronger and diversity-aware baselines, with the oracle ----------------
BASE = [("S1_random", "Random"), ("S10_bm25", "BM25 (lexical)"), ("S2_cosine_full", "Cosine, 384-d"),
        ("S11_mmr_cosine", "MMR, cosine 384-d"), ("S13_nca64_full", "NCA-64 (trained metric)"),
        ("S5_rbf_lda", "RBF, LDA"), ("S7_q_iqp", "Quantum IQP"), ("S12_mmr_iqp", "MMR, quantum IQP"),
        ("S15_rrf_cos_iqp", "Fusion: cosine + IQP"), ("S17_rbf_trained", "RBF, trained scales"),
        ("S16_q_trained", "Quantum IQP, trained scales"), ("S14_oracle", "Label oracle (bound)")]


def tab_baselines():
    T = {(ds, llm): read(f"results/{ds}/runs_{llm}/table1_by_setting.csv") for ds, _, _ in DS for llm in ("llm", "llm2")}
    rows = []
    for sel, name in BASE:
        r = [name]
        for ds, _, D in DS:
            for llm in ("llm", "llm2"):
                t = T[(ds, llm)]; g = t[t.selector == sel] if t is not None else None
                r.append(X if g is None or not len(g) else f(g.acc.mean()))
                if llm == "llm":
                    M[f"{D}Acc{sel.split('_')[0]}"] = r[-1]
                else:
                    M[f"Open{D}Acc{sel.split('_')[0]}"] = r[-1]
        rows.append(r)
    real = bold_best([r for r, (s, _) in zip(rows, BASE) if s != "S14_oracle"], range(1, 7))
    rows = real + [rows[-1]]
    # headroom: oracle minus the best real selector, and minus IQP
    hr = []
    for ds, _, D in DS:
        for llm, L in (("llm", "Main"), ("llm2", "Open")):
            h = (jload(f"results/{ds}/runs_{llm}/verdict.json") or {}).get("headroom") or {}
            M[f"Headroom{D}{L}"] = f(h.get("oracle_minus_best", np.nan))
            M[f"HeadroomIqp{D}{L}"] = f(h.get("oracle_minus_iqp", np.nan))
            best = h.get("best_real")
            M[f"BestReal{D}{L}"] = X if PH or not best else dict(BASE + SEL).get(best, best).replace(", ", " on ")
            hr.append(M[f"Headroom{D}{L}"])
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    body = body.replace(" \\\\\nRBF, LDA", " \\\\\n\\midrule\nRBF, LDA", 1).replace(" \\\\\nLabel oracle", " \\\\\n\\midrule\nLabel oracle", 1)
    tex = r"""\begin{tabular}{@{}l cc cc cc@{}}
\toprule
 & \multicolumn{2}{c}{TREC} & \multicolumn{2}{c}{Emotion} & \multicolumn{2}{c}{DBpedia-14} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Selector & Main & Open & Main & Open & Main & Open \\
\midrule
""" + body + r""" \\
Headroom (oracle $-$ best) & """ + " & ".join(hr) + r""" \\
\bottomrule
\end{tabular}"""
    (OUT / "tab_baselines.tex").write_text(tex + "\n")


# ---------------- Table: pooled gaps, equivalence tests and detectable effects ----------------
PAIR_NAME = {("S7_q_iqp", "S5_rbf_lda"): "IQP $-$ RBF", ("S7_q_iqp", "S3_cosine_lda"): "IQP $-$ Cos (LDA)",
             ("S7_q_iqp", "S2_cosine_full"): "IQP $-$ Cos (384-d)", ("S7_q_iqp", "S13_nca64_full"): "IQP $-$ NCA-64",
             ("S8_q_projected", "S5_rbf_lda"): "Proj. $-$ RBF", ("S12_mmr_iqp", "S11_mmr_cosine"): "MMR-IQP $-$ MMR-Cos",
             ("S15_rrf_cos_iqp", "S2_cosine_full"): "Fusion $-$ Cos (384-d)",
             ("S16_q_trained", "S17_rbf_trained"): "Trained IQP $-$ trained RBF",
             ("S16_q_trained", "S7_q_iqp"): "Trained IQP $-$ IQP"}
FAM = {"primary": "Primary", "secondary": "Secondary", "exploratory": "Exploratory"}


def tab_equiv():
    margin = CFG_DEC.get("equivalence_margin", 2.0)
    lines = []; fam_prev = None
    V = {ds: (jload(f"results/{ds}/runs_llm/verdict.json") or {}).get("pooled", []) for ds, _, _ in DS}
    order = []
    for ds, _, _ in DS:
        for p in V[ds]:
            if (p["family"], p["a"], p["vs"]) not in order:
                order.append((p["family"], p["a"], p["vs"]))
    if not order:
        order = [(fam, a, b) for fam in FAM for (a, b) in CFG_DEC.get(fam, [])]
    for fam, a, b in order:
        if fam_prev is not None and fam != fam_prev:
            lines.append("\\midrule")
        fam_prev = fam
        cells = [FAM.get(fam, fam), PAIR_NAME.get((a, b), f"{a} $-$ {b}")]
        for ds, _, D in DS:
            p = next((p for p in V[ds] if p["a"] == a and p["vs"] == b), None)
            if p is None:
                cells += [X, X, X]; continue
            cells += [f"{sgn(p['gap'])} {ci(p['lo95'], p['hi95'])}",
                      X if PH else (f"{p['p_tost']:.3f}" if p["p_tost"] >= 0.001 else "$<$0.001") + ("$^\\ast$" if p["equivalent"] else ""),
                      f(p["mde"])]
            key = f"{a.split('_')[0]}vs{b.split('_')[0]}{D}"
            M[f"Pooled{key}"], M[f"PooledCI{key}"] = sgn(p["gap"]), ci(p["lo95"], p["hi95"])
            M[f"Tost{key}"] = X if PH else (f"{p['p_tost']:.3f}" if p["p_tost"] >= 0.001 else "<0.001")
            M[f"Mde{key}"] = f(p["mde"])
            M[f"Equiv{key}"] = X if PH else ("equivalent" if p["equivalent"] else "not shown equivalent")
        lines.append(" & ".join(cells) + " \\\\")
    M["EquivMargin"] = f"{margin:g}"
    eq = [p for ds, _, _ in DS for p in V[ds] if p["family"] == "primary" and p["a"] == "S7_q_iqp" and p["vs"] == "S5_rbf_lda"]
    M["NEquivPrimary"] = X if PH or not eq else str(sum(p["equivalent"] for p in eq))
    M["NDatasetsPooled"] = X if PH or not eq else str(len(eq))
    mdes = [p["mde"] for p in eq]
    M["MdeMin"], M["MdeMax"] = (f(min(mdes)), f(max(mdes))) if mdes else (X, X)
    tex = r"""\begin{tabular}{@{}ll ccc ccc ccc@{}}
\toprule
 & & \multicolumn{3}{c}{TREC} & \multicolumn{3}{c}{Emotion} & \multicolumn{3}{c}{DBpedia-14} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
Family & Contrast & Gap [95\% CI] & $p_\mathrm{TOST}$ & MDE & Gap [95\% CI] & $p_\mathrm{TOST}$ & MDE & Gap [95\% CI] & $p_\mathrm{TOST}$ & MDE \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    (OUT / "tab_equiv.tex").write_text(tex + "\n")


# ---------------- Table: robustness to the LLM, the encoder and k ----------------
ROBUST = [("llm", "", "Main LLM", "MiniLM", "1--3"), ("llm2", "", "Qwen2.5-1.5B", "MiniLM", "3"),
          ("llm3", "", "Qwen2.5-7B (4-bit)", "MiniLM", "1"), ("llm2", "__mpnet", "Qwen2.5-1.5B", "MPNet", "1"),
          ("llm2", "__big", "Qwen2.5-1.5B, 2{,}000 test", "MiniLM", "1")]


def tab_robust():
    lines = []
    for llm, suf, lname, enc, seeds in ROBUST:
        cells = [lname, enc]
        for ds, _, D in DS:
            if suf == "__big" and ds == "trec":
                cells += ["--", "--"]; continue                     # TREC's official test set has 500 items
            v = jload(f"results/{ds}{suf}/runs_{llm}/verdict.json")
            p = next((p for p in (v or {}).get("pooled", []) if p["a"] == "S7_q_iqp" and p["vs"] == "S5_rbf_lda"), None)
            if p is None:
                cells += [X, X]; continue
            ns = len(v.get("seeds", []))
            cells += [f"{sgn(p['gap'])} {ci(p['lo95'], p['hi95'])}",
                      X if PH else ("yes" if p["equivalent"] else "no") + (f" ({ns})" if ns else "")]
            tagname = {"llm": "Main", "llm2": "Open", "llm3": "Big"}[llm] + {"": "", "__mpnet": "Mpnet", "__big": "LargeTest"}[suf]
            if suf == "__big":
                M[f"TostLargeTest{D}"] = X if PH else (f"{p['p_tost']:.3f}" if p["p_tost"] >= 0.001 else "<0.001")
                M[f"MdeLargeTest{D}"] = f(p["mde"])
            M[f"RobGap{tagname}{D}"], M[f"RobCI{tagname}{D}"] = sgn(p["gap"]), ci(p["lo95"], p["hi95"])
        lines.append(" & ".join(cells) + " \\\\")
    # k sweep: IQP minus RBF at each k with the open LLM, pooled over datasets that ran it
    ks = []
    for ds, _, D in DS:
        fs = [x for x in glob.glob(f"results/{ds}/runs_llm2/seed0_lda*_k*_bw*.csv") if re.search(r"_bw[\d.]+\.csv$", x)]
        if fs and not PH:
            ks.append(pd.concat([pd.read_csv(x) for x in fs]).assign(ds=ds))
    kline = []
    if ks:
        K = pd.concat(ks); K = K[K.k > 0]
        dmax = K.groupby("dims").k.nunique().idxmax()
        K = K[K.dims == dmax]
        for k in sorted(K.k.unique()):
            w = K[(K.k == k) & K.selector.isin(["S7_q_iqp", "S5_rbf_lda"])].pivot_table(
                index=["ds", "query_id"], columns="selector", values="correct")
            if {"S7_q_iqp", "S5_rbf_lda"} <= set(w.columns):
                gap = (w["S7_q_iqp"] - w["S5_rbf_lda"]).mean() * 100
                M[f"KGap{ {1: 'One', 2: 'Two', 4: 'Four', 8: 'Eight', 16: 'Sixteen', 32: 'ThirtyTwo'}.get(int(k), 'Other')}"] = sgn(gap)
                kline.append(f"$k{{=}}{k}$: {sgn(gap)}")
    M["KSweepGaps"] = X if not kline else "; ".join(kline)
    tex = r"""\begin{tabular}{@{}ll cc cc cc@{}}
\toprule
 & & \multicolumn{2}{c}{TREC} & \multicolumn{2}{c}{Emotion} & \multicolumn{2}{c}{DBpedia-14} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
LLM & Encoder & IQP$-$RBF [95\% CI] & Eq.\ (seeds) & IQP$-$RBF [95\% CI] & Eq.\ (seeds) & IQP$-$RBF [95\% CI] & Eq.\ (seeds) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    (OUT / "tab_robust.tex").write_text(tex + "\n")


# ---------------- Table: why the quantum kernel ranks like RBF (no LLM) ----------------
def tab_geometry():
    """Rank agreement and kernel-target alignment, from the pre-LLM diagnostics; scaling and noise summaries."""
    lines = []
    rows = [("Rank agreement $\\bar\\rho$(IQP, RBF)", lambda P: np.mean([p["rank_agreement"]["S7_q_iqp|S5_rbf_lda"] for p in P]), "RankIqpRbf", 3),
            ("Rank agreement $\\bar\\rho$(IQP, cos LDA)", lambda P: np.mean([p["rank_agreement"]["S7_q_iqp|S3_cosine_lda"] for p in P]), "RankIqpCos", 3),
            ("Rank agreement $\\bar\\rho$(proj., RBF)", lambda P: np.mean([p["rank_agreement"]["S8_q_projected|S5_rbf_lda"] for p in P]), "RankProjRbf", 3),
            ("KTA, IQP", lambda P: np.mean([p["kta"]["S7_q_iqp"] for p in P]), "KtaIqp", 3),
            ("KTA, RBF", lambda P: np.mean([p["kta"]["S5_rbf_lda"] for p in P]), "KtaRbf", 3),
            ("KTA, projected", lambda P: np.mean([p["kta"]["S8_q_projected"] for p in P]), "KtaProj", 3)]
    pre = {}
    for ds, _, _ in DS:
        ps = [json.load(open(p)) for p in sorted(glob.glob(f"results/{ds}/pre_llm_seed*_lda*.json"))]
        pre[ds] = ps if ps and not PH else None
    for name, fn, mac, nd in rows:
        cells = [name]; vals = []
        for ds, _, D in DS:
            try:
                v = fn(pre[ds]) if pre[ds] else np.nan
            except Exception:
                v = np.nan
            cells.append(f(v, nd)); M[f"{mac}{D}"] = f(v, nd); vals.append(v)
        vs = [v for v in vals if not np.isnan(v)]
        M[f"{mac}Min"], M[f"{mac}Max"] = (f(min(vs), nd), f(max(vs), nd)) if vs else (X, X)
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    # scaling: rank agreement and spread at the smallest and largest qubit count (NCA features, tuned c)
    sc_rows = [("$\\bar\\rho$(IQP, RBF), fewest qubits", "rank_agreement", "first", "ScaleRankLo", 3),
               ("$\\bar\\rho$(IQP, RBF), most qubits", "rank_agreement", "last", "ScaleRankHi", 3),
               ("Kernel spread at $c{=}1$, most qubits", "spread_c1", "last", "ScaleSpreadHi", 3),
               ("Kernel spread at tuned $c$, most qubits", "spread_tuned", "last", "ScaleSpreadTunedHi", 3)]
    S = {ds: read(f"results/{ds}/scaling.csv") for ds, _, _ in DS}
    for name, col, where, mac, nd in sc_rows:
        cells = [name]
        for ds, _, D in DS:
            s = S[ds]
            if s is None:
                cells.append(X); M[f"{mac}{D}"] = X; continue
            g = s.groupby("qubits")[col].mean(); v = g.iloc[0] if where == "first" else g.iloc[-1]
            cells.append(f(v, nd)); M[f"{mac}{D}"] = f(v, nd)
        lines.append(" & ".join(cells) + " \\\\")
    for ds, _, D in DS:
        s = S[ds]
        M[f"ScaleQMin{D}"], M[f"ScaleQMax{D}"] = (X, X) if s is None else (str(int(s.qubits.min())), str(int(s.qubits.max())))
    tex = r"""\begin{tabular}{@{}l ccc@{}}
\toprule
Geometry diagnostic (no LLM) & TREC & Emotion & DBpedia-14 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    (OUT / "tab_geometry.tex").write_text(tex + "\n")


def study_numbers():
    """Numbers from the noise study, the bandwidth study and the difficulty strata."""
    n = read("results/trec/noise.csv"); chk = jload("results/trec/noise_checks.json") or {}
    M["NoiseGlobalCheck"] = X if PH or not chk else f"{min(chk.values()):.2f}"
    if n is not None:
        ex = n[n.shots == 0].groupby("p2")[["jaccard_vs_exact", "rank_agreement_vs_exact", "purity", "label_prec"]].mean()
        sh = n[n.shots > 0].groupby("p2")[["jaccard_vs_exact", "label_prec"]].mean()
        p_hi = ex.index.max(); p_mid = min(ex.index, key=lambda p: abs(p - 0.01))
        M["NoisePHi"] = f"{p_hi:g}"; M["NoisePMid"] = f"{p_mid:g}"
        M["NoiseJMid"], M["NoiseJHi"] = f(ex.loc[p_mid, "jaccard_vs_exact"], 2), f(ex.loc[p_hi, "jaccard_vs_exact"], 2)
        M["NoiseRankHi"] = f(ex.loc[p_hi, "rank_agreement_vs_exact"], 3)
        M["NoisePurityHi"] = f(ex.loc[p_hi, "purity"], 2)
        M["NoiseJShotsZero"] = f(sh.loc[sh.index.min(), "jaccard_vs_exact"], 2) if len(sh) else X
        M["NoiseJShotsHi"] = f(sh.loc[p_hi, "jaccard_vs_exact"], 2) if p_hi in sh.index else X
        M["NoisePrecDropHi"] = sgn((ex.loc[p_hi, "label_prec"] - ex.loc[ex.index.min(), "label_prec"]) * 100)
        M["NoiseQubits"] = str(int(n.qubits.iloc[0]))
    b = jload("results/trec/runs_llm2/bandwidth_study.json")
    if b:
        M["BwChanged"] = str(b["n_choices_changed"]); M["BwSettings"] = str(len(b["per_dims"]))
        M["BwMaxGap"] = f(b["max_abs_gap_any_c"]); M["BwGapAtLlmC"] = sgn(b["mean_gap_at_c_llm"])
        M["BwAllWithin"] = "all" if b["all_c_within_margin"] else "not all"
    hw = jload("results/trec/hardware.json")
    if hw:
        M["HwBackend"] = hw["backend"].replace("_", "\\_"); M["HwJ"] = f(hw["jaccard_hw_vs_exact"], 2)
        M["HwRank"] = f(hw["rank_agreement_hw_vs_exact"], 3); M["HwErr"] = f(hw["mean_abs_kernel_error"], 3)
    for ds, _, D in DS:
        d = read(f"results/{ds}/runs_llm/table_difficulty.csv")
        if d is not None:
            d = d.set_index("stratum")
            for s in ("easy", "hard"):
                if s in d.index:
                    M[f"Diff{s.title()}{D}"], M[f"DiffCI{s.title()}{D}"] = sgn(d.loc[s, "gap"]), ci(d.loc[s, "lo"], d.loc[s, "hi"])


# ---------------- Table: the harder dataset (Banking77, 77 intents) ----------------
BANK = [("S1_random", "Random"), ("S10_bm25", "BM25 (lexical)"), ("S2_cosine_full", "Cosine, 384-d"),
        ("S11_mmr_cosine", "MMR, cosine 384-d"), ("S13_nca64_full", "NCA-64 (trained metric)"),
        ("S3_cosine_lda", "Cosine, LDA"), ("S5_rbf_lda", "RBF, LDA"),
        ("S6_q_angle", "Quantum angle"), ("S7_q_iqp", "Quantum IQP"), ("S8_q_projected", "Quantum projected"),
        ("S17_rbf_trained", "RBF, trained scales"), ("S16_q_trained", "Quantum IQP, trained scales"),
        ("S14_oracle", "Label oracle (bound)")]


def tab_banking():
    T = {llm: read(f"results/banking77/runs_{llm}/table1_by_setting.csv") for llm in ("llm2", "llm")}
    rows = []
    for sel, name in BANK:
        r = [name]
        for llm in ("llm2", "llm"):
            t = T[llm]; g = t[t.selector == sel] if t is not None else None
            r.append(X if g is None or not len(g) else f(g.acc.mean()))
            M[f"Bank{'Open' if llm == 'llm2' else 'Main'}Acc{sel.split('_')[0]}"] = r[-1]
        t = T["llm2"]; g = t[t.selector == sel] if t is not None else None
        r.append("--" if sel in ("S11_mmr_cosine", "S12_mmr_iqp", "S14_oracle") else
                 (X if g is None or not len(g) else f(g.ret_auroc.mean())))
        rows.append(r)
    real = bold_best(rows[:-1], [1, 2, 3]); rows = real + [rows[-1]]
    for llm, L in (("llm2", "Open"), ("llm", "Main")):
        v = jload(f"results/banking77/runs_{llm}/verdict.json") or {}
        for a, b, key in (("S7_q_iqp", "S5_rbf_lda", "IqpRbf"), ("S16_q_trained", "S17_rbf_trained", "Trained")):
            p = next((p for p in v.get("pooled", []) if p["a"] == a and p["vs"] == b), None)
            M[f"BankGap{key}{L}"] = sgn(p["gap"]) if p else X
            M[f"BankCI{key}{L}"] = ci(p["lo95"], p["hi95"]) if p else X
            M[f"BankEquiv{key}{L}"] = X if PH or not p else ("equivalent" if p["equivalent"] else "not shown equivalent")
        h = v.get("headroom") or {}
        M[f"BankHeadroom{L}"] = f(h.get("oracle_minus_best", np.nan))
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    body = body.replace(" \\\\\nCosine, LDA", " \\\\\n\\midrule\nCosine, LDA", 1).replace(" \\\\\nLabel oracle", " \\\\\n\\midrule\nLabel oracle", 1)
    tex = r"""\begin{tabular}{@{}l cc c@{}}
\toprule
Selector & Acc (open) & Acc (main) & AUROC$_\mathrm{ret}$ \\
\midrule
""" + body + r""" \\
\bottomrule
\end{tabular}"""
    (OUT / "tab_banking.tex").write_text(tex + "\n")


def qubits_numbers():
    """IQP minus RBF with the open LLM at every qubit count (LDA 2..5, NCA 6..12)."""
    from src.paper_common import qubits_llm, DS4
    worst, best, nca_eq = [], [], []
    for ds, title in DS4:
        q = None if PH else qubits_llm(ds)
        D = re.sub(r"[^A-Za-z]", "", title.replace("-14", ""))
        if q is None:
            M[f"QubitsMaxAbs{D}"] = X; continue
        nca = q[q.method == "NCA"]
        M[f"QubitsMaxAbs{D}"] = f(nca.gap.abs().max())
        best.append(nca.gap.max()); worst.append(nca.gap.min())
        nca_eq.append(int(((nca.lo > -MARGIN) & (nca.hi < MARGIN)).sum())); M[f"QubitsN{D}"] = str(len(nca))
    M["QubitsBestGap"] = sgn(max(best)) if best else X
    M["QubitsWorstGap"] = sgn(min(worst)) if worst else X


# ---------------- numbers quoted in the text ----------------
def text_numbers():
    v = jload("results/trec/runs_llm/verdict.json") or {}
    pbp = v.get("predict_before_pay", {}); rob = v.get("robustness_shots", {})
    M["RhoJaccard"] = f(pbp.get("spearman_absgap_vs_1minus_jaccard", np.nan), 2)
    M["RhoG"] = f(pbp.get("spearman_absgap_vs_g", np.nan), 2)
    M["NSettingsPbp"] = str(pbp["n_settings"]) if pbp and not PH else X
    M["ShotsDiff"] = sgn(rob.get("diff_points", np.nan)); M["ShotsCI"] = ci(*(rob.get("ci") or [np.nan, np.nan]))
    M["ShotsJaccard"] = f(rob.get("jaccard_S7_S9_mean", np.nan), 2)
    pts = []
    for ds, _, _ in DS:
        for llm, _, _ in LLMS:
            t = read(f"results/{ds}/runs_{llm}/table1_by_setting.csv")
            if t is not None:
                pts.append(t)
    if pts:
        P = pd.concat(pts); M["RhoPrecAcc"] = f(spearmanr(P.label_prec, P.acc).correlation, 2)
    else:
        M["RhoPrecAcc"] = X


def write_macros():
    lines = ["% Written by src/paper_tables.py. Do not edit by hand: rerun  python -m src.pipeline paper",
             "\\providecommand{\\res}[1]{#1}"]
    seen = set()
    for k in sorted(M):
        name = re.sub(r"[^A-Za-z]", "", re.sub(r"S(\d+)", lambda m: SHORT["S" + m.group(1)], k))
        if name in seen:
            print(f"warning: macro name clash for {k}; kept the first"); continue
        seen.add(name)
        val = M[k]
        val = "\\textbf{XYZ}" if val == X else val
        lines.append(f"\\newcommand{{\\R{name}}}{{\\res{{{val}}}}}")
    Path("paper/results_macros.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(M)} macros to paper/results_macros.tex")


if __name__ == "__main__":
    for fn in (tab_main, tab_stats, tab_diag, tab_ablation, tab_open, tab_order, tab_baselines, tab_equiv, tab_robust,
               tab_geometry, study_numbers, tab_banking, qubits_numbers, text_numbers):
        try:
            fn()
        except Exception as e:
            print(f"FAILED {fn.__name__}: {e}")
    # every macro the paper uses must exist even if a stage has not run
    for D in ("Trec", "Emotion", "Dbpedia"):
        for L in ("Main", "Open"):
            M.setdefault(f"Gap{D}{L}", X); M.setdefault(f"CI{D}{L}", X); M.setdefault(f"Verdict{D}{L}", X)
    for D in ("Trec", "Emotion", "Dbpedia"):
        for sel, _ in SEL:
            M.setdefault(f"{D}Acc{sel.split('_')[0]}", X)
            M.setdefault(f"Open{D}Auroc{sel.split('_')[0]}", X); M.setdefault(f"Open{D}Ece{sel.split('_')[0]}", X)
            M.setdefault(f"Open{D}Acc{sel.split('_')[0]}", X)
        for mac in ("g", "gMed", "JRbfAngle", "JRbfIqp", "JShots", "RetRbf", "RetIqp", "RetCosFull"):
            M.setdefault(f"{mac}{D}", X)
    for mac in ("g", "gMed", "JRbfAngle", "JRbfIqp", "JShots", "RetRbf", "RetIqp", "RetCosFull"):
        M.setdefault(f"{mac}Min", X); M.setdefault(f"{mac}Max", X)
    for k in list(ABL_MACRO.values()) + ["OpenAurocGapTrec", "OpenAurocGapCITrec", "TrecAccZero", "RhoJaccard", "RhoG",
                                         "NSettingsPbp", "ShotsDiff", "ShotsCI", "ShotsJaccard", "RhoPrecAcc", "OrderSpreadMax"]:
        M.setdefault(k, X)
    for D in ("Trec", "Emotion", "Dbpedia"):
        for sel, _ in BASE:
            M.setdefault(f"{D}Acc{sel.split('_')[0]}", X); M.setdefault(f"Open{D}Acc{sel.split('_')[0]}", X)
        for L in ("Main", "Open"):
            for k in ("Headroom", "HeadroomIqp", "BestReal"):
                M.setdefault(f"{k}{D}{L}", X)
        for fam in ("primary", "secondary", "exploratory"):
            for a, b in CFG_DEC.get(fam, []):
                key = f"{a.split('_')[0]}vs{b.split('_')[0]}{D}"
                for k in ("Pooled", "PooledCI", "Tost", "Mde", "Equiv"):
                    M.setdefault(f"{k}{key}", X)
        for t in ("Main", "Open", "Big", "OpenMpnet"):
            M.setdefault(f"RobGap{t}{D}", X); M.setdefault(f"RobCI{t}{D}", X)
        for mac in ("RankIqpRbf", "RankIqpCos", "RankProjRbf", "KtaIqp", "KtaRbf", "KtaProj", "ScaleRankLo", "ScaleRankHi",
                    "ScaleSpreadHi", "ScaleSpreadTunedHi", "ScaleQMin", "ScaleQMax"):
            M.setdefault(f"{mac}{D}", X)
        for s_ in ("Easy", "Hard"):
            M.setdefault(f"Diff{s_}{D}", X); M.setdefault(f"DiffCI{s_}{D}", X)
    for mac in ("RankIqpRbf", "RankIqpCos", "RankProjRbf", "KtaIqp", "KtaRbf", "KtaProj"):
        M.setdefault(f"{mac}Min", X); M.setdefault(f"{mac}Max", X)
    for k in ["EquivMargin", "NEquivPrimary", "NDatasetsPooled", "MdeMin", "MdeMax", "KSweepGaps", "KGapTwo", "KGapFour",
              "KGapEight", "KGapSixteen", "NoiseGlobalCheck", "NoisePHi", "NoisePMid", "NoiseJMid", "NoiseJHi", "NoiseRankHi",
              "NoisePurityHi", "NoiseJShotsZero", "NoiseJShotsHi", "NoisePrecDropHi", "NoiseQubits", "BwChanged", "BwSettings",
              "BwMaxGap", "BwGapAtLlmC", "BwAllWithin", "HwBackend", "HwJ", "HwRank", "HwErr"]:
        M.setdefault(k, X)
    for sel, _ in BANK:
        for L in ("Open", "Main"):
            M.setdefault(f"Bank{L}Acc{sel.split('_')[0]}", X)
    for L in ("Open", "Main"):
        M.setdefault(f"BankHeadroom{L}", X)
        for key in ("IqpRbf", "Trained"):
            for k_ in ("BankGap", "BankCI", "BankEquiv"):
                M.setdefault(f"{k_}{key}{L}", X)
    for D in ("Trec", "Emotion", "Dbpedia", "Banking"):
        M.setdefault(f"QubitsMaxAbs{D}", X); M.setdefault(f"QubitsN{D}", X)
    for D in ("Trec", "Emotion", "Dbpedia"):
        M.setdefault(f"TostLargeTest{D}", X); M.setdefault(f"MdeLargeTest{D}", X)
        M.setdefault(f"RobGapOpenLargeTest{D}", X); M.setdefault(f"RobCIOpenLargeTest{D}", X)
    for k_ in ("QubitsBestGap", "QubitsWorstGap"):
        M.setdefault(k_, X)
    write_macros(); print("tables in", OUT)
