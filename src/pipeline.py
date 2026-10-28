"""One command per phase. Run from the project folder. Add --dataset (trec, emotion, dbpedia),
--llm (llm = paid main LLM, llm2 = Qwen 1.5B, llm3 = Qwen 7B in 4 bits) and --emb (minilm = main encoder,
mpnet = second encoder) to any stage; the defaults are the config dataset, the main LLM and the main encoder.

  python -m src.pipeline check                     Phase 1   tests + PennyLane check (free)
  python -m src.pipeline features  --dataset trec  Phase 3   download, embed, LDA/NCA (free)
  python -m src.pipeline pre_llm                   Phase 7   tune bandwidth, RBF gamma, projected gamma; overlap; g (free)
  python -m src.pipeline estimate                  Phase 8   number of different prompts and a token estimate (free)
  python -m src.pipeline zeroshot                  Phase 6   500 calls, no examples
  python -m src.pipeline pilot                     Phase 8   all selectors x 100 questions
  python -m src.pipeline main [--seeds 0]          Phase 9   run 1 (seed 0) then run 3 (seeds 1, 2)
  python -m src.pipeline ksweep                    Phase 9   run 2: k = 2 and 8
  python -m src.pipeline nca                       Phase 9   run 4: NCA 6 and 8 dims
  python -m src.pipeline robust                    Phase 9   run 5: IQP exact vs IQP with finite shots (S7 vs S9)
  python -m src.pipeline main --llm llm2 --seeds 0 Phase 9   run 6: the same prompts on the second (free) LLM
  python -m src.pipeline ablate                    ablation study without the LLM: 12 variants (free)
  python -m src.pipeline scaling                   qubit scaling 2 to 12 qubits, no LLM (free)
  python -m src.pipeline noise                     local depolarizing noise + swap-test shots, no LLM (free)
  python -m src.pipeline bwstudy --llm llm2        every bandwidth, tuned by the LLM instead of the vote (free on GPU)
  python -m src.pipeline hardware [--simulate]     OPTIONAL: tiny check on a real IBM quantum computer
  python -m src.pipeline order --llm llm2          ablation: order of the examples in the prompt (free on GPU)
  python -m src.pipeline analyze                   Phases 10 to 12: statistics, verdict, figures (free)
  python -m src.pipeline summary                   one table and figure over all datasets and LLMs (free)
  python -m src.pipeline paper                     every paper table (LaTeX) and figure (PDF) into paper/ (free)

Every LLM answer is cached, so you can stop any stage and run it again without paying twice.
"""
import argparse, json, os, subprocess, sys, glob


def run(*args):
    cmd = [sys.executable, "-m", *map(str, args)]
    print("\n>>>", f"[{os.environ['QK_DATASET']}/{os.environ['QK_LLM']}/{os.environ.get('QK_EMB', 'main encoder')}]",
          " ".join(cmd[2:]), flush=True)
    r = subprocess.run(cmd, env=os.environ.copy())
    if r.returncode != 0:
        sys.exit(f"STOPPED: this step failed: {' '.join(cmd[2:])}")


def load_tuned():
    from src.paths import TUNED
    return json.loads(TUNED.read_text()) if TUNED.exists() else {}


def tuned(key):
    """Arguments that set the tuned bandwidth, RBF gamma and projected gamma for one setting."""
    t = load_tuned()
    if key not in t:
        sys.exit(f"No tuned settings for {key}. Run the pre_llm stage for it first.")
    return ["--bandwidth", t[key]["bandwidth"], "--gamma", t[key]["rbf_gamma"], "--proj-gamma", t[key]["proj_gamma"]]


def settings(method, seeds=None):
    from src.paths import CFG
    dims = CFG["features"]["lda_dims"] if method == "lda" else CFG["features"]["nca_dims"]
    for s in seeds if seeds is not None else CFG["data"]["seeds"]:
        for d in dims:
            yield s, d


def check_llm_config():
    from src.paths import CFG, PROFILE
    c = CFG[PROFILE]
    if c["provider"] != "mock" and (not c["model"] or "PUT_EXACT" in c["model"]):
        sys.exit(f"Set {PROFILE}.model in configs/config.yaml to the exact model name first (Phase 6).")


def stage_pre_llm(method, seeds, backend):
    from src.paths import CFG, RES, TUNED
    t = load_tuned(); k = CFG["experiment"]["k_main"]
    for s, d in settings(method, seeds):
        key = f"seed{s}_{method}{d}"
        common = ["--seed", s, "--method", method, "--dims", d, "--k", k, "--backend", backend]
        run("src.pre_llm_analysis", *common)                    # pass 1: tune on the validation set
        res = json.load(open(RES / f"pre_llm_{key}.json"))
        t[key] = {"bandwidth": float(res["bandwidth"]), "rbf_gamma": float(res["rbf_gamma"]), "proj_gamma": float(res["proj_gamma"])}
        TUNED.write_text(json.dumps(t, indent=2))
        run("src.run_experiment", *common, *tuned(key), "--no-llm")   # pass 2: picks with the tuned values
        run("src.pre_llm_analysis", *common)                    # pass 3: overlap, diversity, vote, label precision
    print("\nTuned settings:", json.dumps(t, indent=2))


def stage_estimate():
    """Counts the prompts that are really different (the cache makes repeats free) for the main run."""
    import pandas as pd
    from src.paths import CFG, DATASET, DATA, RES
    from src.datasets_registry import spec
    from src.llm import build_prompt
    sp = spec(DATASET); wanted = set(CFG["experiment"]["selectors"]); t = load_tuned()
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    unique, chars, rows = set(), 0, 0
    files = glob.glob(str(RES / f"*_k{CFG['experiment']['k_main']}_*_nollm.csv"))
    if not files:
        sys.exit("Run the pre_llm stage first.")
    for f in files:
        df = pd.read_csv(f)
        key = f"seed{df.seed[0]}_{df.method[0]}{df.dims[0]}"
        if key not in t or float(df.bandwidth[0]) != t[key]["bandwidth"] or df.method[0] != "lda":
            continue                                            # skip runs with values we did not choose, and NCA
        df = df[df.selector.isin(wanted)]; rows += len(df)
        pool_idx = pd.read_csv(DATA / f"splits_seed{df.seed.iloc[0]}.csv").pool_idx.dropna().astype(int).values
        for r in df.itertuples():
            ids = tuple(map(int, str(r.picked_ids).split()))
            if (r.seed, r.query_id, ids) not in unique:
                unique.add((r.seed, r.query_id, ids))
                chars += len(build_prompt(test.text[r.query_id], train.text.values[pool_idx][list(ids)],
                                          train.label.values[pool_idx][list(ids)], sp))
    n = len(unique)
    print(f"{DATASET} main run (k = {CFG['experiment']['k_main']}, seeds and dims found): {n:,} different prompts out of {rows:,} rows")
    print(f"About {chars / 4 / 1e6:.2f} million input tokens and {n * 3 / 1e6:.3f} million output tokens (4 characters per token).")
    print("Multiply by your model's price per million tokens (provider pricing page). The second LLM (llm2) runs free on the GPU.")
    print("The k sweep, NCA and robustness runs add roughly 25 to 40 percent on top.")


def stage_llm_runs(kind, seeds, backend="pennylane"):
    from src.paths import CFG
    check_llm_config()
    k = CFG["experiment"]["k_main"]; dmax = CFG["features"]["lda_dims"][-1]
    if kind == "zeroshot":
        run("src.run_experiment", "--seed", 0, "--method", "lda", "--dims", dmax, "--k", 0,
            "--bandwidth", 0.5, "--gamma", 1.0, "--proj-gamma", 1.0, "--selectors", "S1_random", "--backend", "numpy")
    elif kind == "pilot":
        run("src.run_experiment", "--seed", 0, "--method", "lda", "--dims", dmax, "--k", k, *tuned(f"seed0_lda{dmax}"),
            "--n", 100, "--backend", backend)
    elif kind == "main":
        from src.paths import PROFILE
        dims = CFG["experiment"]["llm3_dims"] if PROFILE == "llm3" else CFG["features"]["lda_dims"]
        extra = ["--selectors", ",".join(CFG["experiment"]["llm3_selectors"])] if PROFILE == "llm3" else []
        default = [0] if PROFILE == "llm3" else CFG["data"]["seeds"]          # the 7B model runs seed 0 only
        for s in (seeds if seeds is not None else default):   # seed 0 first for every dim, then the others
            for d in dims:
                run("src.run_experiment", "--seed", s, "--method", "lda", "--dims", d, "--k", k, *tuned(f"seed{s}_lda{d}"),
                    *extra, "--backend", backend)
    elif kind == "ksweep":
        for kk in CFG["experiment"]["k_sweep"]:
            run("src.run_experiment", "--seed", 0, "--method", "lda", "--dims", dmax, "--k", kk, *tuned(f"seed0_lda{dmax}"),
                "--selectors", ",".join(CFG["experiment"]["ksweep_selectors"]), "--backend", backend)
    elif kind == "nca":                     # more qubits WITH the LLM: open LLM all NCA sizes, paid LLM nca_main_dims
        from src.paths import PROFILE
        dims = CFG["experiment"]["nca_main_dims"] if PROFILE == "llm" else CFG["features"]["nca_dims"]
        for d in dims:
            if f"seed0_nca{d}" not in load_tuned():
                sys.exit("Run first:  python -m src.pipeline pre_llm --method nca --seeds 0  (same --dataset)")
            run("src.run_experiment", "--seed", 0, "--method", "nca", "--dims", d, "--k", k, *tuned(f"seed0_nca{d}"),
                "--selectors", ",".join(CFG["experiment"]["nca_selectors"]), "--backend", backend)
    elif kind == "order":
        for o in CFG["experiment"]["order_variants"]:
            run("src.run_experiment", "--seed", 0, "--method", "lda", "--dims", dmax, "--k", k, *tuned(f"seed0_lda{dmax}"),
                "--selectors", ",".join(CFG["experiment"]["order_selectors"]), "--order", o, "--tag", f"order_{o}", "--backend", backend)
    elif kind == "robust":
        run("src.run_experiment", "--seed", 0, "--method", "lda", "--dims", dmax, "--k", k, *tuned(f"seed0_lda{dmax}"),
            "--selectors", ",".join(CFG["experiment"]["robustness_selectors"]), "--tag", "robust", "--backend", backend)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "features", "pre_llm", "estimate", "zeroshot", "pilot",
                                      "main", "ksweep", "nca", "robust", "order", "ablate", "scaling", "noise",
                                      "bwstudy", "hardware", "analyze", "summary", "paper"])
    ap.add_argument("--dataset", default=None, choices=["trec", "emotion", "dbpedia", "banking77"])
    ap.add_argument("--llm", default="llm", choices=["llm", "llm2", "llm3"])
    ap.add_argument("--emb", default=None, choices=["minilm", "mpnet"], help="sentence encoder (default: the main one)")
    ap.add_argument("--big", action="store_true", help="the larger test set (data.big_test_size), its own folders")
    ap.add_argument("--simulate", action="store_true", help="hardware stage: run the circuits on a local simulator")
    ap.add_argument("--method", default="lda", choices=["lda", "nca"])
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 0 or 0,1")
    ap.add_argument("--backend", default="pennylane", choices=["pennylane", "numpy"],
                    help="numpy gives the same kernels faster; use it only after 'check' passes")
    a = ap.parse_args()
    if a.dataset:
        os.environ["QK_DATASET"] = a.dataset
    os.environ["QK_LLM"] = a.llm
    if a.emb:
        os.environ["QK_EMB"] = a.emb
    if a.big:
        os.environ["QK_VARIANT"] = "big"
    from src.paths import CFG, DATASET                          # import after the environment is set
    os.environ["QK_DATASET"] = DATASET
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else None

    if a.stage == "check":
        sys.exit(subprocess.run([sys.executable, "tests/test_core.py"]).returncode)
    if a.stage == "features":
        run("src.prepare_features")
    elif a.stage == "pre_llm":
        stage_pre_llm(a.method, seeds, a.backend)
    elif a.stage == "estimate":
        stage_estimate()
    elif a.stage == "ablate":
        run("src.ablation", "--seeds", a.seeds or "0", "--backend", a.backend)
    elif a.stage == "scaling":
        run("src.scaling_study", "--seed", (seeds or [0])[0])
    elif a.stage == "noise":
        run("src.noise_study", "--seed", (seeds or [0])[0])
    elif a.stage == "bwstudy":
        run("src.bandwidth_study", "--seed", (seeds or [0])[0], "--backend", a.backend)
    elif a.stage == "hardware":
        run("src.hardware_check", "--seed", (seeds or [0])[0], *(["--simulate"] if a.simulate else []))
    elif a.stage in ("zeroshot", "pilot", "main", "ksweep", "nca", "robust", "order"):
        stage_llm_runs(a.stage, seeds, a.backend)
    elif a.stage == "analyze":
        run("src.final_analysis", "--k", CFG["experiment"]["k_main"], "--method", "lda")
        run("src.make_figures")
    elif a.stage == "summary":
        run("src.summarize")
    elif a.stage == "paper":
        run("src.paper_tables")
        run("src.paper_text")
        run("src.paper_figures")


if __name__ == "__main__":
    main()
