"""Smoke test: runs EVERY pipeline stage on fake data for two datasets (TREC with 6 labels, DBpedia with 14),
all three LLM profiles, both encoders and every study, with the NumPy quantum backend and free mock LLMs.
Costs nothing and needs no internet. Run from the project folder:  python tests/smoke_pipeline.py
If it ends with "SMOKE TEST PASSED", the plumbing works. Its numbers mean nothing (fake data, fake LLM)."""
import os, sys, subprocess, shutil, json, numpy as np, pandas as pd, yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from src.data import make_splits, reduce, standardize
from src.datasets_registry import spec

work = "smoke_tmp"; shutil.rmtree(work, ignore_errors=True)
os.makedirs(work + "/configs")
cfg = yaml.safe_load(open("configs/config.yaml"))
cfg["llm"].update(provider="mock", model="mock-main")
cfg["llm2"].update(provider="mock", model="mock-second")
cfg["llm3"].update(provider="mock", model="mock-third", load_in_4bit=False)
cfg["data"].update(pool_size=150, val_size=40); cfg["quantum"]["bandwidths"] = [0.25, 0.5]
cfg["studies"].update(scaling_dims=[2, 3, 6, 9], noise_p2=[0.0, 0.01], noise_dims=[4], hardware_queries=5, hardware_pool=10)
cfg["experiment"]["nca_full_dims"] = 16
cfg["experiment"]["trained_kernel"] = {"subsample": 100, "max_iter": 6}
yaml.safe_dump(cfg, open(work + "/configs/config.yaml", "w"))
os.chdir(work)


def make_fake(name, seed, folder=None, shift=0.0):
    """Fake texts and embeddings with real label names. folder/shift make a second 'encoder' of the same data."""
    labels = spec(name)["labels"]; C = len(labels)
    rng = np.random.default_rng(seed); centers = rng.normal(size=(C, 32)) * 2
    def fake(n):
        y = rng.integers(0, C, n); E = centers[y] + rng.normal(size=(n, 32)) * (3.2 + shift)
        words = ["what", "who", "where", "how", "many", "city", "name", "year", "film", "team"]
        txt = [" ".join(words[j] for j in rng.integers(0, len(words), 5)) + f" {labels[c]}" * int(rng.random() < 0.3)
               for c in y]
        return pd.DataFrame({"text": txt, "label": [labels[i] for i in y]}), E / np.linalg.norm(E, axis=1, keepdims=True)
    train, E_train = fake(700); test, E_test = fake(80)
    D = f"data/{folder or name}"; os.makedirs(D + "/features", exist_ok=True)
    train.to_csv(f"{D}/train.csv", index=False); test.to_csv(f"{D}/test.csv", index=False)
    np.save(f"{D}/features/E_train.npy", E_train); np.save(f"{D}/features/E_test.npy", E_test)
    for s in cfg["data"]["seeds"]:
        pool, val = make_splits(train, s, cfg["data"]["pool_size"], cfg["data"]["val_size"])
        pd.DataFrame({"pool_idx": pd.Series(pool), "val_idx": pd.Series(val)}).to_csv(f"{D}/splits_seed{s}.csv", index=False)
        for method, dims in [("lda", cfg["features"]["lda_dims"]), ("nca", cfg["features"]["nca_dims"])]:
            for d in dims:
                Zp, (Zv, Zq) = reduce(E_train[pool], train.label.values[pool], [E_train[val], E_test], d, method, s)
                Sp, (Sv, Sq) = standardize(Zp, [Zv, Zq]); tag = f"seed{s}_{method}{d}"
                for n, a in [("pool", Zp), ("val", Zv), ("query", Zq), ("pool_std", Sp), ("val_std", Sv), ("query_std", Sq)]:
                    np.save(f"{D}/features/{tag}_{n}.npy", a)


env = dict(os.environ, PYTHONPATH=ROOT)
for k in ("QK_DATASET", "QK_LLM", "QK_EMB"):
    env.pop(k, None)
def stage(*args):
    r = subprocess.run([sys.executable, "-m", "src.pipeline", *args, "--backend", "numpy"], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit("FAILED at stage: " + " ".join(args))
    print(f"ok  pipeline {' '.join(args)}")
    return r.stdout

# Full design on TREC (main LLM)
make_fake("trec", 1)
stage("pre_llm", "--dataset", "trec")
print("   ", stage("estimate", "--dataset", "trec").strip().splitlines()[0])
for st in ["zeroshot", "pilot", "main", "ksweep", "robust"]:
    stage(st, "--dataset", "trec")
stage("pre_llm", "--dataset", "trec", "--method", "nca", "--seeds", "0"); stage("nca", "--dataset", "trec")
out = stage("analyze", "--dataset", "trec")
print("\n".join("    " + l for l in out.splitlines() if l.startswith(("FINAL VERDICT", "checks", "robustness", "predict", "headroom"))))
# Studies without the LLM
for st in ["ablate", "scaling", "noise"]:
    stage(st, "--dataset", "trec")
# Open LLMs: three seeds, k sweep, order, bandwidth study, the 7B model
stage("main", "--dataset", "trec", "--llm", "llm2"); stage("ksweep", "--dataset", "trec", "--llm", "llm2")
stage("order", "--dataset", "trec", "--llm", "llm2"); stage("bwstudy", "--dataset", "trec", "--llm", "llm2")
out2 = stage("analyze", "--dataset", "trec", "--llm", "llm2")
print("\n".join("    " + l for l in out2.splitlines() if l.startswith("ordering")))
stage("main", "--dataset", "trec", "--llm", "llm3"); stage("analyze", "--dataset", "trec", "--llm", "llm3")
stage("nca", "--dataset", "trec", "--llm", "llm2")                    # 6 to 12 qubits with the open LLM
# Second encoder
make_fake("trec", 7, folder="trec__mpnet", shift=0.4)
stage("pre_llm", "--dataset", "trec", "--emb", "mpnet", "--seeds", "0")
stage("main", "--dataset", "trec", "--emb", "mpnet", "--llm", "llm2", "--seeds", "0")
stage("analyze", "--dataset", "trec", "--emb", "mpnet", "--llm", "llm2")

# Secondary dataset, 14 labels
make_fake("dbpedia", 2)
stage("pre_llm", "--dataset", "dbpedia", "--seeds", "0"); stage("main", "--dataset", "dbpedia", "--seeds", "0")
stage("main", "--dataset", "dbpedia", "--llm", "llm2", "--seeds", "0")
for st in ["ablate", "scaling", "analyze"]:
    stage(st, "--dataset", "dbpedia")
stage("analyze", "--dataset", "dbpedia", "--llm", "llm2")
# Larger test set (statistical power), open LLM
make_fake("dbpedia", 4, folder="dbpedia__big")
stage("pre_llm", "--dataset", "dbpedia", "--big", "--seeds", "0")
stage("main", "--dataset", "dbpedia", "--big", "--llm", "llm2", "--seeds", "0")
stage("analyze", "--dataset", "dbpedia", "--big", "--llm", "llm2")

# Harder dataset: Banking77 (77 labels, names read from labels.json)
os.makedirs("data/banking77", exist_ok=True)
fake77 = [f"intent_{w}_{i}" for i, w in enumerate(["card", "transfer", "top_up", "pin", "refund", "exchange", "cash"] * 11)]
json.dump(fake77, open("data/banking77/labels.json", "w"))
make_fake("banking77", 5)
stage("pre_llm", "--dataset", "banking77", "--seeds", "0")
stage("main", "--dataset", "banking77", "--llm", "llm2", "--seeds", "0")
stage("pre_llm", "--dataset", "banking77", "--method", "nca", "--seeds", "0")
stage("nca", "--dataset", "banking77", "--llm", "llm2")
stage("analyze", "--dataset", "banking77", "--llm", "llm2")
print("\n".join("    " + l for l in stage("summary").splitlines()[-8:]))
stage("paper")

need = ["results/trec/runs_llm/verdict.json", "results/trec/runs_llm/disagreements.csv", "results/trec/runs_llm/table_pooled.csv",
        "results/trec/runs_llm/table_difficulty.csv", "results/trec/runs_llm/table_per_class.csv",
        "results/dbpedia/runs_llm/verdict.json", "results/summary_accuracy.csv", "results/trec/tuned.json",
        "results/trec/ablation.csv", "results/trec/figure_material.json", "results/trec/scaling.csv",
        "results/dbpedia/scaling.csv", "results/trec/noise.csv", "results/trec/noise_checks.json",
        "results/trec/runs_llm2/bandwidth_study.json", "results/trec/runs_llm2/table_auroc_comparisons.csv",
        "results/trec/runs_llm3/verdict.json", "results/trec__mpnet/runs_llm2/verdict.json",
        "paper/results_macros.tex", "paper/results_text.tex", "paper/authors.tex", "paper/venue.tex",
        "paper/tables/tab_main.tex", "paper/tables/tab_ablation.tex", "paper/tables/tab_baselines.tex",
        "paper/tables/tab_equiv.tex", "paper/tables/tab_robust.tex",
        "paper/figures/fig_roc.pdf", "paper/figures/fig_acc_qubits.pdf", "paper/figures/fig_concentration_shots.pdf",
        "paper/figures/fig_scaling.pdf", "paper/figures/fig_noise.pdf", "paper/figures/fig_bandwidth.pdf",
        "paper/figures/fig_ksweep.pdf", "paper/figures/fig_difficulty.pdf", "paper/figures/fig_qubits_llm.pdf",
        "paper/tables/tab_banking.tex", "results/banking77/runs_llm2/verdict.json", "results/dbpedia__big/runs_llm2/verdict.json"]
missing = [f for f in need if not os.path.exists(f)]
import glob as _g
assert _g.glob("results/trec/runs_llm2/seed0_nca12_k4_bw*.csv"), "12-qubit open-LLM run missing"
assert not missing, f"missing outputs: {missing}"
v = json.load(open("results/trec/runs_llm/verdict.json"))
assert v["robustness_shots"], "robustness result missing"
assert {"S7_vs_S5", "S7_vs_S3", "S8_vs_S5 (exploratory)", "S7_vs_S2 (secondary)", "S7_vs_S13 (secondary)",
        "S16_vs_S17 (exploratory)"} <= set(v["verdicts"]), v["verdicts"]
assert len(json.load(open("results/trec/runs_llm3/verdict.json"))["seeds"]) == 1, "the 7B model should run seed 0 only"
assert v["headroom"] and v["pooled"] and v["difficulty"], "pooled, headroom or difficulty missing"
assert json.load(open("results/trec/noise_checks.json"))["d4_global_depolarizing_overlap"] == 1.0, "global depolarizing changed picks"
v2 = json.load(open("results/trec/runs_llm2/verdict.json"))
assert v2["ordering"] and v2["reliability"] and "llm_auroc" in v2["mean_metrics"]["S7_q_iqp"], "llm2 extras missing"
assert len(v2["seeds"]) == 3, "open LLM should run three seeds"
assert pd.read_csv("results/dbpedia/runs_llm/table_per_class.csv", index_col=0).shape[1] <= 14
iqp_row = [l for l in open("paper/tables/tab_main.tex") if l.startswith("Quantum IQP")][0]
assert "XYZ" not in iqp_row.split("&")[1], "main table not filled for TREC"
os.chdir(ROOT)
if "--keep" not in sys.argv:
    shutil.rmtree(work)
print("SMOKE TEST PASSED")
