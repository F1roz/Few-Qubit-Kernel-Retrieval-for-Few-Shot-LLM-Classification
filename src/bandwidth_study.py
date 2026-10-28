"""Bandwidth study with an LLM (default: the free open LLM). Answers the reviewer question "would tuning
the bandwidth with the LLM, instead of the neighbour vote, change the verdict?"
For seed 0 and every LDA size it runs the quantum selectors at EVERY bandwidth c, on the validation queries
(to choose c by LLM accuracy) and on the test queries (to report the result at every c). RBF does not depend
on c and is run once per size.
Writes runs_<llm>/bandwidth_study.csv and bandwidth_study.json.
Run:  python -m src.pipeline bwstudy --dataset trec --llm llm2"""
import argparse, json, numpy as np, pandas as pd
from src.paths import CFG, DATASET, PROFILE, RUNS, TUNED, llm_settings
from src import selectors_classical as sc, quantum_kernels as qk, metrics_extra as mx
from src.run_experiment import load_setting, compute
from src.datasets_registry import spec
from src.llm import LLM, build_prompt, parse_answer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=CFG["experiment"]["k_main"]); ap.add_argument("--backend", default="pennylane")
    a = ap.parse_args()
    tuned = json.loads(TUNED.read_text()); llm = LLM(**llm_settings()); sp = spec(DATASET); labels = sp["labels"]
    sels = CFG["studies"]["bandwidth_study_selectors"]; reps = CFG["quantum"]["iqp_repeats"]
    rows = []
    for d in CFG["features"]["lda_dims"]:
        tag = f"seed{a.seed}_lda{d}"
        if tag not in tuned:
            continue
        for split in ("val", "test"):
            ctx = load_setting(a.seed, "lda", d, split=split)
            plan = [(c, sels) for c in CFG["quantum"]["bandwidths"]] + [(None, ["S5_rbf_lda"])]
            for c, wanted in plan:
                cc = c if c is not None else tuned[tag]["bandwidth"]
                picks, _, _ = compute(ctx, wanted, a.k, cc, tuned[tag]["rbf_gamma"], tuned[tag]["proj_gamma"], a.seed,
                                      a.backend, reps, CFG["quantum"]["shots"])
                for sel in wanted:
                    for qi, ids in enumerate(picks[sel]):
                        prompt = build_prompt(ctx["query_text"][qi], ctx["pool_text"][ids], ctx["y_pool"][ids], sp, seed=qi)
                        raw, _, _ = llm.ask(prompt)
                        rows.append({"dims": d, "split": split, "bandwidth": c if c is not None else np.nan, "selector": sel,
                                     "query_id": qi, "correct": parse_answer(raw, labels) == ctx["gold"][qi]})
            print(f"bandwidth study: d={d} {split} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RUNS / "bandwidth_study.csv", index=False)

    # summary: c chosen by the vote vs by the LLM, and the IQP minus RBF gap at every c
    out = {"per_dims": {}, "llm": PROFILE}
    margin = CFG["decision"].get("equivalence_margin", 2.0)
    for d, g in df.groupby("dims"):
        v = g[(g.split == "val") & (g.selector == "S7_q_iqp")].groupby("bandwidth").correct.mean()
        c_llm = float(v[v == v.max()].index.min())
        t = g[g.split == "test"]
        rbf = t[t.selector == "S5_rbf_lda"].sort_values("query_id").correct.values.astype(float)
        per_c = {}
        for c, gc in t[t.selector == "S7_q_iqp"].groupby("bandwidth"):
            q7 = gc.sort_values("query_id").correct.values.astype(float)
            diff, lo, hi = mx.paired_bootstrap_metric(lambda idx: (q7[idx].mean() - rbf[idx].mean()) * 100, len(q7))
            per_c[str(c)] = {"acc_iqp": q7.mean() * 100, "gap": diff, "lo": lo, "hi": hi}
        out["per_dims"][str(d)] = {"c_vote": tuned[f"seed{a.seed}_lda{d}"]["bandwidth"], "c_llm": c_llm,
                                   "acc_rbf": rbf.mean() * 100, "by_c": per_c}
    allg = [x for v in out["per_dims"].values() for x in v["by_c"].values()]
    out["n_choices_changed"] = sum(v["c_vote"] != v["c_llm"] for v in out["per_dims"].values())
    out["max_abs_gap_any_c"] = max(abs(x["gap"]) for x in allg) if allg else None
    out["all_c_within_margin"] = all(-margin < x["lo"] and x["hi"] < margin for x in allg) if allg else None
    llm_gaps = [v["by_c"][str(v["c_llm"])]["gap"] for v in out["per_dims"].values() if str(v["c_llm"]) in v["by_c"]]
    out["mean_gap_at_c_llm"] = float(np.mean(llm_gaps)) if llm_gaps else None
    json.dump(out, open(RUNS / "bandwidth_study.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "per_dims"}, indent=2, default=float))


if __name__ == "__main__":
    main()
