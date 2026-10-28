"""Ablation study without the LLM (free). For one dataset it removes or changes one part of the quantum
selector at a time and measures what happens to the picks on the TEST queries:
  retrieval AUROC (whole ranking), label precision at k, neighbour vote accuracy, vote macro AUROC.
Every variant is compared with the full method (IQP, 2 repeats, tuned bandwidth, LDA, standardized)
by a paired bootstrap over queries.

It also saves what the paper figures need: averaged retrieval ROC curves, kernel value samples per
bandwidth (concentration), and the effect of the number of shots.
Run:  python -m src.pipeline ablate --dataset trec          (seed 0, every LDA size)"""
import argparse, json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from src.paths import CFG, DATASET, DATA, FEAT, RES, TUNED
from src.datasets_registry import spec
from src.data import standardize
from src import selectors_classical as sc, quantum_kernels as qk, metrics_extra as mx


def metrics(S, k, y_pool, gold, labels):
    picked = sc.top_k_from_scores(S, k)
    return {"ret_auroc": mx.retrieval_auroc(S, y_pool, gold),
            "label_prec": sc.label_precision(picked, y_pool, gold),
            "vote_correct": (sc.neighbour_vote(picked, y_pool) == gold).astype(float),
            "vote_P": mx.soft_vote(picked, y_pool, labels), "picked": picked}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0"); ap.add_argument("--k", type=int, default=CFG["experiment"]["k_main"])
    ap.add_argument("--backend", default="pennylane")
    a = ap.parse_args()
    states = qk.pl_states if a.backend == "pennylane" else qk.np_states
    labels = spec(DATASET)["labels"]; tuned = json.loads(TUNED.read_text())
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv"); gold = test.label.values
    E_tr, E_te = np.load(FEAT / "E_train.npy"), np.load(FEAT / "E_test.npy")
    rows, extra, perq = [], {}, {}
    dims_all = CFG["features"]["lda_dims"]
    for seed in map(int, a.seeds.split(",")):
        pool_idx = pd.read_csv(DATA / f"splits_seed{seed}.csv").pool_idx.dropna().astype(int).values
        y_pool = train.label.values[pool_idx]
        for d in dims_all:
            tag = f"seed{seed}_lda{d}"
            if tag not in tuned:
                continue
            t = tuned[tag]; c, g, pg = t["bandwidth"], t["rbf_gamma"], t["proj_gamma"]
            L = lambda n: np.load(FEAT / f"{tag}_{n}.npy")
            Zp, Zq, Sp, Sq = L("pool"), L("query"), L("pool_std"), L("query_std")
            reps = CFG["quantum"]["iqp_repeats"]
            Iq, Ip = states(Sq * c, "iqp", reps), states(Sp * c, "iqp", reps)
            K_full = qk.kernel_from_states(Iq, Ip)

            # PCA and truncation see the same pool and the same number of qubits as LDA
            pca = PCA(n_components=d, random_state=seed).fit(E_tr[pool_idx])
            Pp, (Pq,) = standardize(pca.transform(E_tr[pool_idx]), [pca.transform(E_te)])
            Tp, (Tq,) = standardize(E_tr[pool_idx][:, :d], [E_te[:, :d]])
            def iqp(Q, P, cc=c, r=reps):
                return qk.kernel_from_states(states(Q * cc, "iqp", r), states(P * cc, "iqp", r))

            variants = {   # name: (group, scores)
                "Full method (IQP, 2 reps, tuned c, LDA)": ("reference", K_full),
                "No entanglement (angle map)": ("feature map", qk.kernel_from_states(states(Sq * c, "angle", reps), states(Sp * c, "angle", reps))),
                "IQP, 1 repeat": ("feature map", iqp(Sq, Sp, r=1)),
                "IQP, 3 repeats": ("feature map", iqp(Sq, Sp, r=3)),
                "Projected kernel instead of fidelity": ("kernel type", qk.projected_kernel(qk.bloch_vectors(Iq), qk.bloch_vectors(Ip), pg)),
                "No bandwidth tuning (c = 1)": ("scaling", iqp(Sq, Sp, cc=1.0)),
                "No standardization": ("scaling", iqp(Zq, Zp)),
                "PCA instead of LDA": ("features", iqp(Pq, Pp)),
                "First d embedding values": ("features", iqp(Tq, Tp)),
                "1,024 shots instead of exact": ("hardware", qk.shot_kernel(K_full, 1024, seed=1000 + seed)),
                "RBF on the same features (classical)": ("classical", sc.rbf_scores(Sq, Sp, g)),
                "RBF on PCA features (classical)": ("classical", sc.rbf_scores(Pq, Pp, sc.median_gamma(Pp))),
                "Cosine on the same features (classical)": ("classical", sc.cosine_scores(Zq, Zp)),
            }
            ref = metrics(K_full, a.k, y_pool, gold, labels)
            for name, (group, S) in variants.items():
                m = ref if group == "reference" else metrics(S, a.k, y_pool, gold, labels)
                row = {"dataset": DATASET, "seed": seed, "dims": d, "variant": name, "group": group,
                       "ret_auroc": np.nanmean(m["ret_auroc"]), "label_prec": m["label_prec"].mean(),
                       "vote_acc": m["vote_correct"].mean(), "vote_macro_auroc": mx.macro_auroc(m["vote_P"], gold, labels),
                       "jaccard_vs_full": float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(m["picked"], ref["picked"])]))}
                for key in ("ret_auroc", "label_prec"):
                    perq.setdefault((seed, name, key), []).append(m[key] - ref[key])
                    diff = lambda idx, key=key: np.nanmean(m[key][idx]) - np.nanmean(ref[key][idx])
                    row[f"d_{key}"], row[f"d_{key}_lo"], row[f"d_{key}_hi"] = mx.paired_bootstrap_metric(diff, len(gold), seed=seed)
                rows.append(row)

            if seed == 0 and d == dims_all[-1]:      # figure material at the largest LDA size
                sel = {"S1_random": np.random.default_rng(0).random(K_full.shape), "S3_cosine_lda": sc.cosine_scores(Zq, Zp),
                       "S5_rbf_lda": sc.rbf_scores(Sq, Sp, g), "S6_q_angle": variants["No entanglement (angle map)"][1],
                       "S7_q_iqp": K_full, "S8_q_projected": variants["Projected kernel instead of fidelity"][1]}
                roc = {}
                for s_name, S in sel.items():
                    fpr, tpr = mx.mean_roc_curve(S, y_pool, gold)
                    roc[s_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auroc": float(np.nanmean(mx.retrieval_auroc(S, y_pool, gold)))}
                extra["roc"] = roc
                rng = np.random.default_rng(0); conc = {}
                iu = np.triu_indices(len(Sp), 1); pick = rng.choice(len(iu[0]), 4000, replace=False)
                for bw in CFG["quantum"]["bandwidths"]:
                    St = states(Sp * bw, "iqp", reps); Kpp = qk.kernel_from_states(St, St)
                    conc[f"IQP c={bw}"] = Kpp[iu][pick].tolist()
                Kr = sc.rbf_scores(Sp, Sp, g); conc["RBF"] = Kr[iu][pick].tolist()
                extra["concentration"] = conc
                shots = {}
                for N in [64, 128, 256, 512, 1024, 2048, 4096, 16384]:
                    vals = [metrics(qk.shot_kernel(K_full, N, seed=r), a.k, y_pool, gold, labels) for r in range(5)]
                    shots[N] = {"label_prec": float(np.mean([v["label_prec"].mean() for v in vals])),
                                "label_prec_sd": float(np.std([v["label_prec"].mean() for v in vals])),
                                "jaccard_vs_exact": float(np.mean([np.mean([len(set(x) & set(y)) / len(set(x) | set(y))
                                                                            for x, y in zip(v["picked"], ref["picked"])]) for v in vals]))}
                extra["shots"] = {"exact_label_prec": float(ref["label_prec"].mean()), "by_shots": shots}
                extra["setting"] = tag
            print("ablation done:", tag, flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "ablation.csv", index=False)
    # pooled over sizes d: average each query's change over d, then bootstrap over queries (valid interval)
    pooled = {}
    for (seed, name, key), arrs in perq.items():
        D = np.nanmean(np.vstack(arrs), axis=0)
        est, lo, hi = mx.paired_bootstrap_metric(lambda idx: np.nanmean(D[idx]), len(D), seed=seed)
        r = pooled.setdefault((seed, name), {"dataset": DATASET, "seed": seed, "variant": name, "n_dims": len(arrs)})
        r[f"d_{key}"], r[f"d_{key}_lo"], r[f"d_{key}_hi"] = est, lo, hi
    pd.DataFrame(list(pooled.values())).to_csv(RES / "ablation_pooled.csv", index=False)
    json.dump(extra, open(RES / "figure_material.json", "w"))
    show = df.groupby(["group", "variant"])[["ret_auroc", "label_prec", "vote_acc", "d_ret_auroc", "d_label_prec"]].mean()
    print(show.round(4).to_string()); print("saved", RES / "ablation.csv", "and", RES / "figure_material.json")


if __name__ == "__main__":
    main()
