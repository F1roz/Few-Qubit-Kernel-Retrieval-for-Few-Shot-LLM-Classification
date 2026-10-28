"""Hardware-noise study (no LLM, free). How far do the IQP picks move under realistic noise?
For the tuned setting (seed 0, LDA d from config studies.noise_dims) it simulates density matrices of the IQP
map with LOCAL depolarizing noise after every gate layer (two-qubit error p2, one-qubit error p2/10),
estimates the kernel with a swap test, with exact probabilities and with a finite number of shots,
and compares with the noiseless picks.

It also checks the ranking argument of the paper: GLOBAL depolarizing noise, K -> (1 - p) K + p / 2^d,
must leave every pick unchanged (overlap 1.0) when no shots are involved.
Run:  python -m src.pipeline noise --dataset trec"""
import argparse, json, numpy as np, pandas as pd
from src.paths import CFG, DATASET, DATA, FEAT, RES, TUNED
from src import selectors_classical as sc, quantum_kernels as qk, metrics_extra as mx


def topk_jaccard(A, B):
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(A, B)]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=CFG["experiment"]["k_main"])
    a = ap.parse_args()
    tuned = json.loads(TUNED.read_text()); reps = CFG["quantum"]["iqp_repeats"]
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    pool_idx = pd.read_csv(DATA / f"splits_seed{a.seed}.csv").pool_idx.dropna().astype(int).values
    y_pool, gold = train.label.values[pool_idx], test.label.values
    rows, checks = [], {}
    for d in CFG["studies"]["noise_dims"]:
        tag = f"seed{a.seed}_lda{d}"; c = tuned[tag]["bandwidth"]
        Sp, Sq = np.load(FEAT / f"{tag}_pool_std.npy") * c, np.load(FEAT / f"{tag}_query_std.npy") * c
        K0 = qk.kernel_from_states(qk.np_states(Sq, "iqp", reps), qk.np_states(Sp, "iqp", reps))
        P0 = sc.top_k_from_scores(K0, a.k)
        Kg = qk.global_depolarize(K0, 0.1, d)
        checks[f"d{d}_global_depolarizing_overlap"] = topk_jaccard(sc.top_k_from_scores(Kg, a.k), P0)
        for p2 in CFG["studies"]["noise_p2"]:
            p1 = p2 / 10
            Rp = [qk.np_iqp_density(x, reps, p1, p2) for x in Sp]
            Rq = [qk.np_iqp_density(x, reps, p1, p2) for x in Sq]
            Kn = qk.density_kernel(Rq, Rp)
            purity = float(np.mean([np.real(np.trace(r @ r)) for r in Rp[:50]]))
            for shots in [None] + list(CFG["studies"]["noise_shots"]):
                for rep in range(1 if shots is None else 5):
                    K = Kn if shots is None else qk.swap_test_shots(Kn, shots, seed=rep)
                    P = sc.top_k_from_scores(K, a.k)
                    rows.append({"dataset": DATASET, "qubits": d, "p2": p2, "p1": p1, "shots": shots or 0, "rep": rep,
                                 "purity": purity, "jaccard_vs_exact": topk_jaccard(P, P0),
                                 "rank_agreement_vs_exact": mx.rank_agreement(K, K0),
                                 "label_prec": float(sc.label_precision(P, y_pool, gold).mean()),
                                 "ret_auroc": float(np.nanmean(mx.retrieval_auroc(K, y_pool, gold)))})
            last = [r for r in rows if r["p2"] == p2 and r["qubits"] == d]
            print(f"d={d} p2={p2}: purity {purity:.3f}, J exact-prob {last[0]['jaccard_vs_exact']:.3f}, "
                  f"J with shots {np.mean([r['jaccard_vs_exact'] for r in last[1:]]):.3f}", flush=True)
    df = pd.DataFrame(rows); df.to_csv(RES / "noise.csv", index=False)
    json.dump(checks, open(RES / "noise_checks.json", "w"), indent=2)
    print("global depolarizing check (must be 1.0):", checks); print("saved", RES / "noise.csv")


if __name__ == "__main__":
    main()
