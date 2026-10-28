"""Phase 3 building blocks: splits, embeddings, supervised reduction, scaling."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis

LABELS = ["abbreviation", "entity", "description", "human", "location", "number"]   # TREC, kept for tests


def make_splits(train_df, seed, pool_size=500, val_size=100):
    """Stratified pool + separate validation set, both from the TRAIN split only."""
    rng = np.random.default_rng(seed)
    frac = (pool_size + val_size) / len(train_df)
    picked = (train_df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(max(2, round(len(g) * frac)), random_state=seed)))
    idx = rng.permutation(picked.index.to_numpy())
    return np.sort(idx[:pool_size]), np.sort(idx[pool_size:pool_size + val_size])


def embed(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(list(texts), normalize_embeddings=True, show_progress_bar=True)


def reduce(E_pool, y_pool, E_other_list, dims, method="lda", seed=0):
    """Fit on the POOL ONLY, then transform the other sets. Never fit on queries."""
    if method == "lda":
        red = LinearDiscriminantAnalysis(n_components=dims)
    else:
        red = NeighborhoodComponentsAnalysis(n_components=dims, random_state=seed)
    Z_pool = red.fit_transform(E_pool, y_pool)
    return Z_pool, [red.transform(E) for E in E_other_list]


def standardize(Z_pool, Z_other_list):
    """Scale each feature with the POOL mean and std (needed before quantum encoding)."""
    mu, sd = Z_pool.mean(0), Z_pool.std(0) + 1e-9
    return (Z_pool - mu) / sd, [(Z - mu) / sd for Z in Z_other_list]
