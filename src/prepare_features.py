"""Phase 3 runner: builds every feature file for ONE dataset.
Run:  python -m src.pipeline features --dataset trec     (or emotion, dbpedia)"""
import numpy as np, pandas as pd
import json
from src.paths import CFG, DATASET, DATA, FEAT, EMB_MODEL, VARIANT
from src.data import make_splits, embed, reduce, standardize
from src.datasets_registry import load, stratified_sample, spec, SPECS

c = CFG["data"]
train, test = load(DATASET)
if SPECS[DATASET]["labels"] is None:                        # label names come from the data (Banking77)
    (DATA / "labels.json").write_text(json.dumps(sorted(set(train.label))))
assert set(train.label) <= set(spec(DATASET)["labels"]), "label words do not match the registry"
if DATASET != "trec":                                       # TREC is already small; keep it whole
    train = stratified_sample(train, c["train_cap"], seed=12345)
n_test = c["big_test_size"] if VARIANT == "big" else c["test_size"]
if VARIANT == "big" and len(test) <= c["test_size"]:
    raise SystemExit(f"{DATASET} has only {len(test)} test items: the --big variant adds nothing here.")
test = stratified_sample(test, n_test, seed=12345)
train.to_csv(DATA / "train.csv", index=False); test.to_csv(DATA / "test.csv", index=False)
print(DATASET, "train", train.shape, "test", test.shape)
print(train.label.value_counts())

E_train = embed(train.text, EMB_MODEL); np.save(FEAT / "E_train.npy", E_train)
E_test = embed(test.text, EMB_MODEL); np.save(FEAT / "E_test.npy", E_test)

for s in c["seeds"]:
    pool_idx, val_idx = make_splits(train, s, c["pool_size"], c["val_size"])
    assert not set(pool_idx) & set(val_idx), "pool and validation overlap!"
    pd.DataFrame({"pool_idx": pd.Series(pool_idx), "val_idx": pd.Series(val_idx)}).to_csv(DATA / f"splits_seed{s}.csv", index=False)
    y_pool = train.label.values[pool_idx]
    for method, dims_list in [("lda", CFG["features"]["lda_dims"]), ("nca", CFG["features"]["nca_dims"])]:
        for d in dims_list:
            Zp, (Zv, Zq) = reduce(E_train[pool_idx], y_pool, [E_train[val_idx], E_test], d, method, s)
            Sp, (Sv, Sq) = standardize(Zp, [Zv, Zq])
            tag = f"seed{s}_{method}{d}"
            for name, arr in [("pool", Zp), ("val", Zv), ("query", Zq), ("pool_std", Sp), ("val_std", Sv), ("query_std", Sq)]:
                np.save(FEAT / f"{tag}_{name}.npy", arr)
            print("saved", tag, Zp.shape, Zq.shape)
print("DONE: features ready for", DATASET)
