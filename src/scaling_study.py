"""Qubit scaling study (no LLM, free): where does the IQP kernel start to ORDER the pool differently from
RBF, and where does it collapse? For d = 2 .. 12 qubits (NCA features fitted on the pool, so the LDA limit of
C - 1 dimensions does not apply) it tunes the bandwidth c on the validation set exactly as the main runs do,
then measures on the test queries:
  g(RBF || IQP)          geometric difference, smallest over the RBF gamma grid
  rank agreement         mean Spearman correlation of each query's row, IQP vs RBF
  J(RBF, IQP)            overlap of the top-k picks
  spread                 standard deviation of off-diagonal IQP values on the pool (concentration), at the tuned c and at c = 1
  KTA                    kernel-target alignment of IQP and RBF on the pool
  retrieval AUROC        IQP, RBF, projected
  label precision@k      IQP, RBF
The NumPy simulator is used (it matches PennyLane to 1e-6 in tests/test_core.py) because 12-qubit
PennyLane runs are slow. Run:  python -m src.pipeline scaling --dataset trec"""
import argparse, numpy as np, pandas as pd
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from src.paths import CFG, DATASET, DATA, FEAT, RES
from src.data import standardize
from src import selectors_classical as sc, quantum_kernels as qk, metrics_extra as mx


def topk_jaccard(A, B):
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(A, B)]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=CFG["experiment"]["k_main"]); ap.add_argument("--backend", default="numpy")
    a = ap.parse_args()
    states = qk.pl_states if a.backend == "pennylane" else qk.np_states
    reps, ridge = CFG["quantum"]["iqp_repeats"], CFG["quantum"]["ridge"]
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    sp = pd.read_csv(DATA / f"splits_seed{a.seed}.csv")
    pool_idx = sp.pool_idx.dropna().astype(int).values; val_idx = sp.val_idx.dropna().astype(int).values
    E_tr, E_te = np.load(FEAT / "E_train.npy"), np.load(FEAT / "E_test.npy")
    Ep, Ev = E_tr[pool_idx], E_tr[val_idx]
    y_pool, y_val, gold = train.label.values[pool_idx], train.label.values[val_idx], test.label.values
    rows = []
    for d in CFG["studies"]["scaling_dims"]:
        red = NeighborhoodComponentsAnalysis(n_components=d, random_state=a.seed, max_iter=100).fit(Ep, y_pool)
        Sp, (Sv, Sq) = standardize(red.transform(Ep), [red.transform(Ev), red.transform(E_te)])
        # tune c on validation, as in the main runs
        best, best_acc, cache = None, -1, {}
        for c in CFG["quantum"]["bandwidths"]:
            Ip = states(Sp * c, "iqp", reps); Iv = states(Sv * c, "iqp", reps); cache[c] = Ip
            acc = float(np.mean(sc.neighbour_vote(sc.top_k_from_scores(qk.kernel_from_states(Iv, Ip), a.k), y_pool) == y_val))
            if acc > best_acc:
                best, best_acc = c, acc
        Ip = cache[best]; Iq = states(Sq * best, "iqp", reps)
        Kqp, Kpp = qk.kernel_from_states(Iq, Ip), qk.kernel_from_states(Ip, Ip)
        g0 = sc.median_gamma(Sp)
        g_min = min(qk.geometric_difference(Kpp, sc.rbf_scores(Sp, Sp, g0 * m), ridge) for m in CFG["classical"]["rbf_gamma_multipliers"])
        Rqp, Rpp = sc.rbf_scores(Sq, Sp, g0), sc.rbf_scores(Sp, Sp, g0)
        Bq, Bp = qk.bloch_vectors(Iq), qk.bloch_vectors(Ip)
        Pqp = qk.projected_kernel(Bq, Bp, qk.projected_median_gamma(Bp))
        iu = np.triu_indices(len(Sp), 1)
        I1 = cache.get(1.0) if 1.0 in cache else states(Sp * 1.0, "iqp", reps)
        K1 = qk.kernel_from_states(I1, I1)
        pick_q, pick_r = sc.top_k_from_scores(Kqp, a.k), sc.top_k_from_scores(Rqp, a.k)
        rows.append({"dataset": DATASET, "seed": a.seed, "qubits": d, "bandwidth": best,
                     "g_rbf_iqp": g_min, "rank_agreement": mx.rank_agreement(Kqp, Rqp), "jaccard_rbf_iqp": topk_jaccard(pick_q, pick_r),
                     "spread_tuned": float(Kpp[iu].std()), "mean_tuned": float(Kpp[iu].mean()),
                     "spread_c1": float(K1[iu].std()), "mean_c1": float(K1[iu].mean()),
                     "kta_iqp": mx.kernel_target_alignment(Kpp, y_pool), "kta_rbf": mx.kernel_target_alignment(Rpp, y_pool),
                     "ret_auroc_iqp": float(np.nanmean(mx.retrieval_auroc(Kqp, y_pool, gold))),
                     "ret_auroc_rbf": float(np.nanmean(mx.retrieval_auroc(Rqp, y_pool, gold))),
                     "ret_auroc_proj": float(np.nanmean(mx.retrieval_auroc(Pqp, y_pool, gold))),
                     "prec_iqp": float(sc.label_precision(pick_q, y_pool, gold).mean()),
                     "prec_rbf": float(sc.label_precision(pick_r, y_pool, gold).mean())})
        print(f"d={d}: c={best}, g={g_min:.2f}, rank={rows[-1]['rank_agreement']:.3f}, J={rows[-1]['jaccard_rbf_iqp']:.2f}, "
              f"spread(c=1)={rows[-1]['spread_c1']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "scaling.csv", index=False)
    print("saved", RES / "scaling.csv")


if __name__ == "__main__":
    main()
