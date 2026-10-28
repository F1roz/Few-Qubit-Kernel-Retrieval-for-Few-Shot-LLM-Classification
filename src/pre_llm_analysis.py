"""Phase 7: tune on the validation set, then measure everything we can without the LLM.
  1) tune the quantum bandwidth c (shared by S6 to S9) by the IQP neighbour vote accuracy on VALIDATION.
     RBF gamma and projected gamma do not change any retrieval (rankings are monotone in distance);
     gamma only enters the geometric difference, where we report the minimum over the gamma grid
  2) geometric difference, kernel concentration
  3) if the no LLM run exists: overlap (Jaccard), diversity, vote accuracy, label precision
Run:  python -m src.pre_llm_analysis --seed 0 --method lda --dims 5"""
import argparse, json, numpy as np, pandas as pd
from src.paths import CFG, DATA, FEAT, RES
from src import selectors_classical as sc, quantum_kernels as qk, metrics_extra as mx
from src.evaluate import jaccard_sets, diversity

PROJ_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0]    # same 4 candidates as the RBF gamma


def vote_acc(K, y_pool, y_val, k):
    return float(np.mean(sc.neighbour_vote(sc.top_k_from_scores(K, k), y_pool) == y_val))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--method", default="lda")
    ap.add_argument("--dims", type=int, default=5); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--backend", default="pennylane")
    a = ap.parse_args()
    tag = f"seed{a.seed}_{a.method}{a.dims}"; reps = CFG["quantum"]["iqp_repeats"]
    train = pd.read_csv(DATA / "train.csv"); sp = pd.read_csv(DATA / f"splits_seed{a.seed}.csv")
    pool_idx = sp.pool_idx.dropna().astype(int).values; val_idx = sp.val_idx.dropna().astype(int).values
    Sp, Sv, Zp = np.load(FEAT / f"{tag}_pool_std.npy"), np.load(FEAT / f"{tag}_val_std.npy"), np.load(FEAT / f"{tag}_pool.npy")
    y_pool, y_val = train.label.values[pool_idx], train.label.values[val_idx]
    states = qk.pl_states if a.backend == "pennylane" else qk.np_states

    # 1a) quantum bandwidth (ties go to the first, i.e. smaller, value)
    bw_report, cache = {}, {}
    for bw in CFG["quantum"]["bandwidths"]:
        Iv, Ip = states(Sv * bw, "iqp", reps), states(Sp * bw, "iqp", reps)
        cache[bw] = (Iv, Ip); Kpp = qk.kernel_from_states(Ip, Ip)
        bw_report[bw] = {"val_vote_acc": vote_acc(qk.kernel_from_states(Iv, Ip), y_pool, y_val, a.k), **qk.concentration_stats(Kpp)}
    bw = max(bw_report, key=lambda b: bw_report[b]["val_vote_acc"])
    Iv, Ip = cache[bw]

    # 1b) RBF gamma. For ONE query an RBF kernel ranks the pool by distance whatever gamma is, so gamma never
    #     changes which examples S5 retrieves. Gamma only matters for the geometric difference g, where we give
    #     the classical side the benefit of the doubt: g is the MINIMUM over the gamma grid (closest classical kernel).
    g0 = sc.median_gamma(Sp); rbf_report = {}

    # 1c) projected kernel gamma: also rank-invariant for one query (S8 retrieval depends only on c).
    #     It is kept at the median rule; it only enters g(RBF || projected).
    Bv, Bp = qk.bloch_vectors(Iv), qk.bloch_vectors(Ip); p0 = qk.projected_median_gamma(Bp); proj_report = {}
    proj_gamma = float(p0)

    # 2) geometric difference: can the quantum kernels behave differently from classical ones on this pool?
    Kq = qk.kernel_from_states(Ip, Ip); Kproj = qk.projected_kernel(Bp, Bp, proj_gamma)
    Kcos = sc.cosine_scores(Zp, Zp) ** 2          # cos^2 is positive semidefinite (max(cos, 0)^2 need not be)
    r = CFG["quantum"]["ridge"]
    for m in CFG["classical"]["rbf_gamma_multipliers"]:
        Krbf = sc.rbf_scores(Sp, Sp, g0 * m)
        rbf_report[m] = {"gamma": float(g0 * m), "g_quantum_vs_rbf": qk.geometric_difference(Kq, Krbf, r),
                         "g_projected_vs_rbf": qk.geometric_difference(Kproj, Krbf, r)}
    best = min(rbf_report, key=lambda m: rbf_report[m]["g_quantum_vs_rbf"])
    rbf_gamma = rbf_report[best]["gamma"]
    out = {"bandwidth": bw, "rbf_gamma": rbf_gamma, "proj_gamma": proj_gamma,
           "g_quantum_vs_rbf": rbf_report[best]["g_quantum_vs_rbf"],
           "g_quantum_vs_rbf_median": rbf_report[min(rbf_report, key=lambda m: abs(m - 1.0))]["g_quantum_vs_rbf"],
           "g_quantum_vs_cosine": qk.geometric_difference(Kq, Kcos, r),
           "g_projected_vs_rbf": min(v["g_projected_vs_rbf"] for v in rbf_report.values()),
           "bandwidth_report": bw_report, "rbf_gamma_report": rbf_report, "proj_gamma_report": proj_report}
    # retrieval only uses each query's ORDER of the pool: compare kernels by rank agreement on validation queries
    Kv_iqp = qk.kernel_from_states(Iv, Ip); Kv_rbf = sc.rbf_scores(Sv, Sp, rbf_gamma)
    Kv_proj = qk.projected_kernel(Bv, Bp, proj_gamma); Kv_cos = sc.cosine_scores(np.load(FEAT / f"{tag}_val.npy"), Zp)
    out["rank_agreement"] = {"S7_q_iqp|S5_rbf_lda": mx.rank_agreement(Kv_iqp, Kv_rbf),
                             "S7_q_iqp|S3_cosine_lda": mx.rank_agreement(Kv_iqp, Kv_cos),
                             "S8_q_projected|S5_rbf_lda": mx.rank_agreement(Kv_proj, Kv_rbf)}
    # how well each pool kernel lines up with the labels (kernel-target alignment)
    out["kta"] = {"S7_q_iqp": mx.kernel_target_alignment(Kq, y_pool),
                  "S5_rbf_lda": mx.kernel_target_alignment(sc.rbf_scores(Sp, Sp, rbf_gamma), y_pool),
                  "S8_q_projected": mx.kernel_target_alignment(Kproj, y_pool),
                  "S3_cosine_lda": mx.kernel_target_alignment(Kcos, y_pool)}

    # 3) tables from the no LLM run with these exact settings
    f = RES / f"{tag}_k{a.k}_bw{bw}_nollm.csv"
    if f.exists():
        res = pd.read_csv(f)
        if np.isclose(res.rbf_gamma.iloc[0], rbf_gamma) and np.isclose(res.proj_gamma.iloc[0], proj_gamma):
            picks = {s: [list(map(int, x.split())) for x in g.picked_ids] for s, g in res.groupby("selector")}
            names = sorted(picks)
            out["jaccard"] = {f"{x}|{y}": jaccard_sets(picks[x], picks[y]) for i, x in enumerate(names) for y in names[i + 1:]}
            out["diversity"] = {s: diversity(picks[s], y_pool, Zp) for s in names}
            out["vote_accuracy"] = res.groupby("selector").vote_correct.mean().to_dict()
            out["label_precision"] = res.groupby("selector").label_prec.mean().to_dict()
    json.dump(out, open(RES / f"pre_llm_{tag}.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("_report")}, indent=2, default=float))


if __name__ == "__main__":
    main()
