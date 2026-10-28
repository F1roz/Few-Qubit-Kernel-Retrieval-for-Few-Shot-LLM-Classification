"""Run:  python -m pytest -q tests   (or: python tests/test_core.py)
Every test here must pass before the main run."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import quantum_kernels as qk, selectors_classical as sc
from src.evaluate import paired_bootstrap_diff, mcnemar_exact, holm, jaccard_sets, judge_one_setting
from src.llm import build_prompt, parse_answer
from src.datasets_registry import spec, SPECS

rng = np.random.default_rng(0)


def test_angle_kernel_equals_closed_form():
    """Proves the 'classical in disguise' claim for Map A (Contribution C2)."""
    A, B = rng.normal(size=(6, 4)), rng.normal(size=(5, 4))
    K = qk.kernel_from_states(qk.np_states(A, "angle"), qk.np_states(B, "angle"))
    assert np.max(np.abs(K - qk.angle_kernel_closed_form(A, B))) < 1e-9


def test_states_are_normalized_and_diagonal_is_one():
    X = rng.normal(size=(5, 4))
    for fmap in ["angle", "iqp"]:
        S = qk.np_states(X, fmap)
        assert np.allclose(np.linalg.norm(S, axis=1), 1)
        assert np.allclose(np.diag(qk.kernel_from_states(S, S)), 1)


def test_iqp_is_not_the_angle_kernel():
    X = rng.normal(size=(8, 4))
    Ka = qk.kernel_from_states(qk.np_states(X, "angle"), qk.np_states(X, "angle"))
    Ki = qk.kernel_from_states(qk.np_states(X, "iqp"), qk.np_states(X, "iqp"))
    assert np.abs(Ka - Ki).max() > 0.05


def test_pennylane_matches_numpy_if_installed():
    try:
        import pennylane  # noqa
    except ImportError:
        print("  (PennyLane not installed here: skipped)"); return
    X = rng.normal(size=(5, 4))
    for fmap in ["angle", "iqp"]:
        Kp = qk.kernel_from_states(qk.pl_states(X, fmap), qk.pl_states(X, fmap))
        Kn = qk.kernel_from_states(qk.np_states(X, fmap), qk.np_states(X, fmap))
        assert np.abs(Kp - Kn).max() < 1e-6, fmap


def test_geometric_difference_is_about_one_for_same_kernel():
    X = rng.normal(size=(40, 3))
    K = sc.rbf_scores(X, X, 0.5)
    assert 0.9 < qk.geometric_difference(K, K) <= 1.0001


def test_selectors_on_toy_points():
    P = np.array([[0, 0], [1, 0], [5, 5], [0, 1]], float); Q = np.array([[0.9, 0.1]])
    assert list(sc.top_k_from_scores(sc.euclid_scores(Q, P), 2)[0]) == [1, 0]
    assert sc.top_k_from_scores(sc.cosine_scores(Q, P), 1)[0, 0] == 1
    assert list(sc.neighbour_vote(np.array([[2, 0, 1]]), np.array(["a", "b", "c"]))) == ["c"]   # tie: most similar wins


def test_statistics():
    a = np.array([1] * 60 + [0] * 40); b = np.array([1] * 50 + [0] * 50)
    diff, lo, hi = paired_bootstrap_diff(a, b)
    assert abs(diff - 10) < 1e-9 and lo <= diff <= hi
    assert mcnemar_exact(a, a) == 1.0
    assert list(holm([0.01, 0.04, 0.03])) == [0.03, 0.06, 0.06]
    assert jaccard_sets([[1, 2]], [[2, 3]]) == 1 / 3
    assert judge_one_setting(3.0, 1.0, 5.0, 0.01, 2, 0.05) == "quantum better"
    assert judge_one_setting(0.2, -1.5, 1.8, 0.8, 2, 0.05) == "no practical difference"


def test_prompt_and_parser():
    sp = spec("trec")
    p = build_prompt("Who wrote Hamlet?", ["Most similar?", "Second?"], ["human", "entity"], sp)
    assert p.index("Second?") < p.index("Most similar?") < p.index("Who wrote Hamlet?")
    assert parse_answer(" Human.", sp["labels"]) == "human" and parse_answer("I think LOC", sp["labels"]) == "invalid"
    d = spec("dbpedia")
    assert len(d["labels"]) == 14 and parse_answer("Film", d["labels"]) == "film"
    for name in [n for n in SPECS if SPECS[n]["labels"] is not None]:   # every label must be found by the parser
        labs = spec(name)["labels"]
        assert all(parse_answer(l, labs) == l for l in labs), name


def test_bloch_vectors_of_known_states():
    """RY(t)|0> has Bloch vector (sin t, 0, cos t). One qubit angle map with x = t gives exactly that."""
    t = np.array([[0.3], [1.2], [-0.7]])
    B = qk.bloch_vectors(qk.np_states(t, "angle"))
    assert np.allclose(B, np.c_[np.sin(t[:, 0]), np.zeros(3), np.cos(t[:, 0])], atol=1e-9)
    X = rng.normal(size=(6, 4)); B = qk.bloch_vectors(qk.np_states(X, "iqp"))
    assert B.shape == (6, 12)
    assert np.all(np.linalg.norm(B.reshape(6, 4, 3), axis=2) <= 1 + 1e-9)   # reduced states lie inside the ball


def test_projected_kernel_and_shots():
    X = rng.normal(size=(7, 3)); B = qk.bloch_vectors(qk.np_states(X, "iqp"))
    Kp = qk.projected_kernel(B, B, qk.projected_median_gamma(B))
    assert np.allclose(np.diag(Kp), 1) and np.allclose(Kp, Kp.T) and Kp.min() > 0
    K = qk.kernel_from_states(qk.np_states(X, "iqp"), qk.np_states(X, "iqp"))
    Ks = qk.shot_kernel(K, 1024, seed=0)
    assert np.abs(Ks - K).max() < 0.1 and np.all((Ks >= 0) & (Ks <= 1))
    assert np.array_equal(Ks, qk.shot_kernel(K, 1024, seed=0))               # same seed, same noise
    assert np.abs(qk.shot_kernel(K, 10**7, seed=0) - K).max() < 0.002        # many shots -> exact kernel


def test_label_precision():
    prec = sc.label_precision(np.array([[0, 1], [2, 2]]), np.array(["a", "b", "a"]), np.array(["a", "a"]))
    assert list(prec) == [0.5, 1.0]


def test_classical_baselines():
    S = sc.bm25_scores(["who wrote hamlet"], ["hamlet was written by shakespeare", "paris is in france"])
    assert S[0, 0] > 0 and S[0, 1] == 0                                      # shared words score, others do not
    rel = np.array([[0.9, 0.85, 0.1]]); P = np.array([[1, 0.99, 0.0], [0.99, 1, 0.0], [0.0, 0.0, 1]])
    assert list(sc.mmr_select(rel, P, 2, lam=0.5)[0]) == [0, 2]               # the near-duplicate is skipped
    assert list(sc.mmr_select(rel, P, 2, lam=1.0)[0]) == [0, 1]               # lam = 1 is plain top-k
    o = sc.oracle_select(np.array([[0.9, 0.8, 0.7, 0.6]]), np.array(list("abab")), ["b"], 3)
    assert list(o[0]) == [1, 3, 0]                                            # gold-label items first, then the rest
    F = sc.rrf_scores([np.array([[3.0, 2.0, 1.0]]), np.array([[1.0, 2.0, 3.0]])])
    assert np.allclose(F[0, 0], F[0, 2])                                      # symmetric ranks tie


def test_noise_models():
    X = rng.normal(size=(5, 3)) * 0.5
    R = [qk.np_iqp_density(x) for x in X]
    assert np.abs(qk.density_kernel(R, R) - qk.kernel_from_states(qk.np_states(X), qk.np_states(X))).max() < 1e-10
    Rn = [qk.np_iqp_density(x, p1=0.01, p2=0.05) for x in X]
    for r in Rn:                                                              # valid density matrices, less pure
        assert abs(np.trace(r).real - 1) < 1e-10 and np.allclose(r, r.conj().T) and np.linalg.eigvalsh(r).min() > -1e-10
        assert np.trace(r @ r).real < 1
    K = qk.kernel_from_states(qk.np_states(X), qk.np_states(rng.normal(size=(7, 3))))
    assert (sc.top_k_from_scores(qk.global_depolarize(K, 0.2, 3), 3) == sc.top_k_from_scores(K, 3)).all()   # ranking kept
    assert np.abs(qk.swap_test_shots(K, 10**7) - K).max() < 0.01


def test_retrieval_statistics():
    from src import metrics_extra as mx
    A = rng.random((4, 9))
    assert abs(mx.rank_agreement(A, A) - 1) < 1e-12 and abs(mx.rank_agreement(A, -A) + 1) < 1e-12
    y = np.array(list("aabb")); assert abs(mx.kernel_target_alignment((y[:, None] == y[None]).astype(float), y) - 1) < 1e-12
    p, lo, hi = mx.tost_bootstrap(np.zeros(200) + rng.normal(0, 0.1, 200), 2.0)
    assert p < 0.05 and -2 < lo < hi < 2                                      # clearly equivalent
    p2_, _, _ = mx.tost_bootstrap(np.zeros(200) + 5 + rng.normal(0, 0.1, 200), 2.0)
    assert p2_ > 0.5                                                          # clearly not equivalent



def test_trained_kernels_and_parser():
    from src import trained_kernels as tk
    from src.llm import parse_answer
    y = np.repeat([0, 1, 2], 20)
    X = rng.normal(size=(60, 3)); X[:, 0] += y * 2.0                   # only feature 0 carries the label
    for kind in ("rbf", "iqp"):
        out = tk.fit_scales(kind, X, y, 0.5, seed=0, subsample=60, max_iter=25)
        assert out["alignment"] >= out["alignment_init"] - 1e-9, kind      # training never makes alignment worse
        K = tk.kernel(kind, X[:5], X[:5], np.array(out["scales"]))
        assert np.allclose(np.diag(K), 1, atol=1e-9)
    w = np.array(tk.fit_scales("rbf", X, y, 0.5, seed=0, subsample=60, max_iter=40)["scales"])
    assert w[0] > max(w[1], w[2]), w                                    # the informative feature gets the largest scale
    L = ["card_payment", "card_payment_fee_charged", "exchange_rate"]
    assert parse_answer("card_payment_fee_charged", L) == "card_payment_fee_charged"
    assert parse_answer("Exchange rate.", L) == "exchange_rate"
    assert parse_answer("card_payment_fee", L) == "card_payment"        # longest label the reply starts with
    assert parse_answer("Human", ["abbreviation", "human"]) == "human" and parse_answer("maybe", L) == "invalid"


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("PASS", name)
