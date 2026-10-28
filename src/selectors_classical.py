"""Phase 4: classical selectors, neighbour vote, label precision. Owner: classical role.
Every selector returns, for each query, k pool indices ordered MOST similar first."""
import numpy as np


def top_k_from_scores(S, k):
    """S: (n_queries, n_pool) scores, larger = more similar. Ties go to the smaller pool index."""
    order = np.argsort(-S, axis=1, kind="stable")
    return order[:, :k]


def cosine_scores(Q, P):
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    return Qn @ Pn.T


def euclid_scores(Q, P):
    d2 = (Q ** 2).sum(1)[:, None] + (P ** 2).sum(1)[None, :] - 2 * Q @ P.T
    return -np.sqrt(np.maximum(d2, 0))          # negative distance: larger = closer


def rbf_scores(Q, P, gamma):
    d = -euclid_scores(Q, P)
    return np.exp(-gamma * d ** 2)


def median_gamma(P):
    """Starting value for the RBF gamma: 1 / median squared distance between pool points."""
    d = -euclid_scores(P, P)
    return 1.0 / np.median(d[np.triu_indices_from(d, 1)] ** 2)


def random_select(n_queries, n_pool, k, seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(n_pool, k, replace=False) for _ in range(n_queries)]).reshape(n_queries, k)


def neighbour_vote(picked, y_pool):
    """Majority label of the k picked examples. Ties go to the most similar example's label."""
    preds = []
    for row in picked:
        if len(row) == 0:
            preds.append("none"); continue
        labels = [y_pool[i] for i in row]
        counts = {l: labels.count(l) for l in labels}
        best = max(counts.values())
        preds.append(next(l for l in labels if counts[l] == best))
    return np.array(preds)


def label_precision(picked, y_pool, gold):
    """Share of the k picked examples whose label equals the query's true label (precision at k)."""
    return np.array([np.mean([y_pool[i] == g for i in row]) if len(row) else np.nan for row, g in zip(picked, gold)])


# ---------------- stronger classical baselines (version 5) ----------------
import re as _re


def _tokens(t):
    return _re.findall(r"[a-z0-9]+", str(t).lower())


def bm25_scores(query_texts, pool_texts, k1=1.5, b=0.75):
    """Okapi BM25 (Robertson and Zaragoza 2009): keyword retrieval with no embedding at all."""
    docs = [_tokens(t) for t in pool_texts]
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
    tf = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        for w in d:
            tf[i, vocab[w]] += 1
    dl = tf.sum(1); avg = dl.mean() if len(dl) else 1.0
    df = (tf > 0).sum(0); idf = np.log((len(docs) - df + 0.5) / (df + 0.5) + 1.0)
    norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[:, None] / avg))          # pool x vocab
    S = np.zeros((len(query_texts), len(docs)))
    for qi, q in enumerate(query_texts):
        ids = [vocab[w] for w in set(_tokens(q)) if w in vocab]
        if ids:
            S[qi] = norm[:, ids] @ idf[ids]
    return S


def mmr_select(S_qp, S_pp, k, lam=0.5):
    """Maximal marginal relevance (Carbonell and Goldstein 1998): each next example maximizes
    lam * similarity to the query - (1 - lam) * highest similarity to an example already chosen.
    Returns (n_queries, k) indices, in the order chosen (most relevant first)."""
    out = np.zeros((len(S_qp), k), dtype=int)
    for qi, rel in enumerate(S_qp):
        chosen = [int(np.argmax(rel))]
        red = S_pp[chosen[0]].copy()
        while len(chosen) < k:
            score = lam * rel - (1 - lam) * red
            score[chosen] = -np.inf
            j = int(np.argmax(score)); chosen.append(j)
            red = np.maximum(red, S_pp[j])
        out[qi] = chosen
    return out


def rrf_scores(score_list, c=60):
    """Reciprocal rank fusion (Cormack, Clarke and Buettcher 2009): sum over rankers of 1 / (c + rank)."""
    total = 0
    for S in score_list:
        ranks = np.argsort(np.argsort(-S, axis=1, kind="stable"), axis=1) + 1
        total = total + 1.0 / (c + ranks)
    return total


def oracle_select(S, y_pool, gold, k):
    """UPPER BOUND, not a selector: the k most similar pool items that carry the query's GOLD label
    (topped up with the most similar others if fewer than k exist). It uses the answer, so it only
    measures how much any selector could gain."""
    order = np.argsort(-S, axis=1, kind="stable"); y_pool = np.asarray(y_pool)
    out = np.zeros((len(S), k), dtype=int)
    for qi, (row, g) in enumerate(zip(order, gold)):
        same = row[y_pool[row] == g][:k]
        rest = [i for i in row if i not in set(same)][:k - len(same)]
        out[qi] = np.r_[same, rest].astype(int)
    return out
