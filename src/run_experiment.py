"""Runs the selectors for one setting and (optionally) calls the LLM. One CSV row per query and selector.
Usually started by src.pipeline. Direct use:
  python -m src.run_experiment --seed 0 --dims 5 --k 4 --bandwidth 0.5 --gamma 0.1 --proj-gamma 0.5 --no-llm
"""
import argparse, time, numpy as np, pandas as pd
from src.paths import CFG, DATASET, PROFILE, EMB, DATA, FEAT, RES, RUNS, llm_settings
from src import selectors_classical as sc, quantum_kernels as qk
from src.datasets_registry import spec
from src.llm import LLM, build_prompt, parse_answer
from src import metrics_extra as mx

ALL = ["S1_random", "S2_cosine_full", "S3_cosine_lda", "S4_euclid_lda", "S5_rbf_lda",
       "S6_q_angle", "S7_q_iqp", "S8_q_projected", "S9_q_iqp_shots",
       "S10_bm25", "S11_mmr_cosine", "S12_mmr_iqp", "S13_nca64_full", "S14_oracle", "S15_rrf_cos_iqp",
       "S16_q_trained", "S17_rbf_trained"]
QUANTUM_NEEDS = {"S6_q_angle": "angle", "S7_q_iqp": "iqp", "S8_q_projected": "iqp", "S9_q_iqp_shots": "iqp",
                 "S12_mmr_iqp": "iqp", "S15_rrf_cos_iqp": "iqp", "S16_q_trained": "iqp"}
NO_RANKING = {"S11_mmr_cosine", "S12_mmr_iqp", "S14_oracle"}   # picked by a rule, not by one score per item


def nca_full(seed, E_pool, y_pool, E_query_all, dims):
    """S13 features: NCA fitted on the pool labels of the FULL embedding (a trained metric retriever).
    Cached per seed, because it does not depend on the LDA size."""
    fp, fq = FEAT / f"seed{seed}_nca{dims}full_pool.npy", FEAT / f"seed{seed}_nca{dims}full_query.npy"
    if fp.exists() and fq.exists():
        return np.load(fp), np.load(fq)
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    n = min(dims, E_pool.shape[1], len(E_pool) - 1)
    red = NeighborhoodComponentsAnalysis(n_components=n, random_state=seed, max_iter=100).fit(E_pool, y_pool)
    Np, Nq = red.transform(E_pool), red.transform(E_query_all)
    np.save(fp, Np); np.save(fq, Nq)
    return Np, Nq


def compute(ctx, wanted, k, bandwidth, gamma, proj_gamma, seed, backend="pennylane", reps=2, shots=1024):
    """Picks for every wanted selector. ctx holds the arrays of one setting (see load_setting).
    Returns picks {selector: (n_queries, k)}, scores {selector: matrix or None}, kernel times {selector: s}."""
    c = ctx; scores, picks, times = {}, {}, {}
    need = set(wanted)
    if need & {"S11_mmr_cosine", "S14_oracle", "S15_rrf_cos_iqp"}:
        need.add("S2_cosine_full")
    if need & {"S12_mmr_iqp", "S15_rrf_cos_iqp", "S9_q_iqp_shots", "S8_q_projected"}:
        need.add("S7_q_iqp")
    if "S1_random" in need:
        scores["S1_random"] = np.random.default_rng(seed).random((len(c["Zq"]), len(c["Zp"])))
    if "S2_cosine_full" in need:
        scores["S2_cosine_full"] = sc.cosine_scores(c["Eq"], c["Ep"])
    if "S3_cosine_lda" in need:
        scores["S3_cosine_lda"] = sc.cosine_scores(c["Zq"], c["Zp"])
    if "S4_euclid_lda" in need:
        scores["S4_euclid_lda"] = sc.euclid_scores(c["Zq"], c["Zp"])
    if "S5_rbf_lda" in need:
        scores["S5_rbf_lda"] = sc.rbf_scores(c["Sq"], c["Sp"], gamma)
    states = qk.pl_states if backend == "pennylane" else qk.np_states
    if "S6_q_angle" in need:
        t0 = time.time(); Aq, Ap = states(c["Sq"] * bandwidth, "angle", reps), states(c["Sp"] * bandwidth, "angle", reps)
        scores["S6_q_angle"] = qk.kernel_from_states(Aq, Ap); times["S6_q_angle"] = time.time() - t0
    if "S7_q_iqp" in need:
        t0 = time.time(); Iq, Ip = states(c["Sq"] * bandwidth, "iqp", reps), states(c["Sp"] * bandwidth, "iqp", reps)
        scores["S7_q_iqp"] = qk.kernel_from_states(Iq, Ip); times["S7_q_iqp"] = time.time() - t0
        if "S8_q_projected" in need:
            t0 = time.time(); scores["S8_q_projected"] = qk.projected_kernel(qk.bloch_vectors(Iq), qk.bloch_vectors(Ip), proj_gamma)
            times["S8_q_projected"] = time.time() - t0
        if "S12_mmr_iqp" in need:
            Kpp = qk.kernel_from_states(Ip, Ip)
    if "S9_q_iqp_shots" in need:
        scores["S9_q_iqp_shots"] = qk.shot_kernel(scores["S7_q_iqp"], shots, seed=1000 + seed)
    if "S10_bm25" in need:
        scores["S10_bm25"] = sc.bm25_scores(c["query_text"], c["pool_text"])
    if "S13_nca64_full" in need:
        scores["S13_nca64_full"] = sc.euclid_scores(c["Nq"], c["Np"])
    if need & {"S16_q_trained", "S17_rbf_trained"}:
        from src import trained_kernels as tk
        tcfg = CFG["experiment"].get("trained_kernel", {})
        if "S16_q_trained" in need:
            w = tk.get_scales("iqp", FEAT, c["tag"], c["Sp"], c["y_pool"], bandwidth, seed, tcfg, reps)
            t0 = time.time()                                  # kernel time only (training is cached and one-off)
            scores["S16_q_trained"] = tk.kernel("iqp", c["Sq"], c["Sp"], w, reps, states)
            times["S16_q_trained"] = time.time() - t0
        if "S17_rbf_trained" in need:
            w = tk.get_scales("rbf", FEAT, c["tag"], c["Sp"], c["y_pool"], float(np.sqrt(sc.median_gamma(c["Sp"]))), seed, tcfg, reps)
            scores["S17_rbf_trained"] = tk.kernel("rbf", c["Sq"], c["Sp"], w)
    if "S15_rrf_cos_iqp" in need:
        scores["S15_rrf_cos_iqp"] = sc.rrf_scores([scores["S2_cosine_full"], scores["S7_q_iqp"]])
    for s_, S in scores.items():
        picks[s_] = sc.top_k_from_scores(S, k)
    lam = CFG["experiment"].get("mmr_lambda", 0.5)
    if "S11_mmr_cosine" in need and k > 0:
        picks["S11_mmr_cosine"] = sc.mmr_select(scores["S2_cosine_full"], sc.cosine_scores(c["Ep"], c["Ep"]), k, lam)
    if "S12_mmr_iqp" in need and k > 0:
        picks["S12_mmr_iqp"] = sc.mmr_select(scores["S7_q_iqp"], Kpp, k, lam)
    if "S14_oracle" in need and k > 0:
        picks["S14_oracle"] = sc.oracle_select(scores["S2_cosine_full"], c["y_pool"], c["gold"], k)
    for s_ in NO_RANKING:
        if s_ in need:
            scores[s_] = None
            picks.setdefault(s_, np.zeros((len(c["Zq"]), 0), dtype=int))
    return picks, scores, times


def load_setting(seed, method, dims, n=None, split="test"):
    """Everything one setting needs, as a dict. split = test (the queries) or val (validation queries)."""
    tag = f"seed{seed}_{method}{dims}"
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    sp = pd.read_csv(DATA / f"splits_seed{seed}.csv")
    pool_idx = sp.pool_idx.dropna().astype(int).values; val_idx = sp.val_idx.dropna().astype(int).values
    L = lambda nm: np.load(FEAT / f"{tag}_{nm}.npy")
    E_tr, E_te = np.load(FEAT / "E_train.npy"), np.load(FEAT / "E_test.npy")
    y_pool = train.label.values[pool_idx]
    if split == "val":
        queries = train.iloc[val_idx].reset_index(drop=True); Eq = E_tr[val_idx]; Zq, Sq = L("val"), L("val_std")
    else:
        queries = test; Eq = E_te; Zq, Sq = L("query"), L("query_std")
    ctx = dict(tag=tag, train=train, queries=queries, pool_idx=pool_idx, y_pool=y_pool,
               pool_text=train.text.values[pool_idx], Zp=L("pool"), Sp=L("pool_std"), Ep=E_tr[pool_idx],
               Zq=Zq, Sq=Sq, Eq=Eq)
    nd = CFG["experiment"].get("nca_full_dims", 64)
    if split == "val":
        Np, _ = nca_full(seed, ctx["Ep"], y_pool, E_te, nd)
        ctx["Nq"] = _nca_transform_val(seed, ctx["Ep"], y_pool, Eq, nd)
    else:
        Np, Nq = nca_full(seed, ctx["Ep"], y_pool, E_te, nd); ctx["Nq"] = Nq
    ctx["Np"] = Np
    if n:
        for key in ("Zq", "Sq", "Eq", "Nq"):
            ctx[key] = ctx[key][:n]
        ctx["queries"] = ctx["queries"].iloc[:n]
    ctx["query_text"] = ctx["queries"].text.values; ctx["gold"] = ctx["queries"].label.values
    return ctx


def _nca_transform_val(seed, E_pool, y_pool, E_val, dims):
    fv = FEAT / f"seed{seed}_nca{dims}full_val.npy"
    if fv.exists():
        return np.load(fv)
    from sklearn.neighbors import NeighborhoodComponentsAnalysis
    n = min(dims, E_pool.shape[1], len(E_pool) - 1)
    red = NeighborhoodComponentsAnalysis(n_components=n, random_state=seed, max_iter=100).fit(E_pool, y_pool)
    V = red.transform(E_val); np.save(fv, V)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--method", default="lda")
    ap.add_argument("--dims", type=int, default=5); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bandwidth", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=None, help="RBF gamma (default: median rule)")
    ap.add_argument("--proj-gamma", type=float, default=None, help="projected kernel gamma (default: median rule)")
    ap.add_argument("--n", type=int, default=None, help="only the first n queries (pilot)")
    ap.add_argument("--no-llm", action="store_true"); ap.add_argument("--backend", default="pennylane")
    ap.add_argument("--selectors", default=None, help="comma list; default = experiment.selectors in config")
    ap.add_argument("--tag", default="", help="extra word added to the file name, e.g. robust")
    ap.add_argument("--order", default="near_last", choices=["near_last", "near_first", "shuffled"],
                    help="example order in the prompt (only the ordering ablation changes it)")
    a = ap.parse_args()

    ctx = load_setting(a.seed, a.method, a.dims, n=a.n)
    test, tag = ctx["queries"], ctx["tag"]
    gamma = a.gamma or sc.median_gamma(ctx["Sp"])
    if a.proj_gamma is None:
        st = qk.pl_states if a.backend == "pennylane" else qk.np_states
        proj_gamma = qk.projected_median_gamma(qk.bloch_vectors(st(ctx["Sp"] * a.bandwidth, "iqp", CFG["quantum"]["iqp_repeats"])))
    else:
        proj_gamma = a.proj_gamma
    wanted = a.selectors.split(",") if a.selectors else (ALL if a.no_llm else CFG["experiment"]["selectors"])
    picked, scores, times = compute(ctx, wanted, a.k, a.bandwidth, gamma, proj_gamma, a.seed, a.backend,
                                    CFG["quantum"]["iqp_repeats"], CFG["quantum"]["shots"])
    y_pool, pool_text = ctx["y_pool"], ctx["pool_text"]
    sp = spec(DATASET)
    llm = None if a.no_llm else LLM(**llm_settings())
    labels = sp["labels"]
    want_probs = bool(llm) and CFG[PROFILE].get("score_labels", False) and \
        a.seed in CFG[PROFILE].get("score_label_seeds", [a.seed]) and \
        len(labels) <= CFG["experiment"].get("probs_max_labels", 20)

    rows = []
    for sel in wanted:
        vote = sc.neighbour_vote(picked[sel], y_pool)
        prec = sc.label_precision(picked[sel], y_pool, test.label.values)
        ret = mx.retrieval_auroc(scores[sel], y_pool, test.label.values) if a.k > 0 and scores[sel] is not None \
            else np.full(len(test), np.nan)
        vp = mx.probs_to_str(mx.soft_vote(picked[sel], y_pool, labels)) if a.k > 0 else [""] * len(test)
        for qi in range(len(test)):
            ids = picked[sel][qi]
            row = dict(dataset=DATASET, encoder=EMB, llm=PROFILE, seed=a.seed, method=a.method, dims=a.dims, k=a.k,
                       bandwidth=a.bandwidth, rbf_gamma=gamma, proj_gamma=proj_gamma, selector=sel,
                       query_id=qi, picked_ids=" ".join(map(str, ids)), gold=test.label.values[qi],
                       vote=vote[qi], label_prec=prec[qi], ret_auroc=ret[qi], vote_p=vp[qi], order=a.order)
            if llm:
                prompt = build_prompt(test.text.values[qi], pool_text[ids], y_pool[ids], sp, order=a.order, seed=qi)
                raw, cached, ms = llm.ask(prompt)
                row.update(raw_answer=raw, pred=parse_answer(raw, labels), cached=cached, latency_ms=ms)
                if want_probs:
                    row["llm_p"] = " ".join(f"{x:.4f}" for x in llm.label_probs(prompt, labels))
            rows.append(row)
            if llm and len(rows) % 250 == 0:
                print(f"  {len(rows)} of {len(wanted) * len(test)} prompts done", flush=True)
    df = pd.DataFrame(rows)
    df["vote_correct"] = df.vote == df.gold
    df["kernel_s"] = df.selector.map(times).fillna(0.0)          # seconds to build this selector's kernel
    if llm: df["correct"] = df.pred == df.gold
    folder = RES if a.no_llm else RUNS
    name = f"{tag}_k{a.k}_bw{a.bandwidth}{'_' + a.tag if a.tag else ''}{'_nollm' if a.no_llm else ''}{'_n' + str(a.n) if a.n else ''}.csv"
    df.to_csv(folder / name, index=False)
    print("saved", folder / name, "| kernel seconds:", {k: round(v, 2) for k, v in times.items()})
    cols = ["vote_correct", "label_prec", "ret_auroc"] + (["correct"] if llm else [])
    print(df.groupby("selector")[cols].mean().round(3))


if __name__ == "__main__":
    main()
