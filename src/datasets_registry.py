"""The four datasets. Each loader returns (train, test) DataFrames with columns text, label,
where label is a plain lowercase word the LLM can answer with."""
import json, re
from pathlib import Path
import pandas as pd

SPECS = {
    "trec": {
        "labels": ["abbreviation", "entity", "description", "human", "location", "number"],
        "item": "Question",
        "instruction": "Classify each question by the type of answer it asks for. Use exactly one of these labels: {labels}. Reply with the label word only.",
    },
    "emotion": {
        "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
        "item": "Text",
        "instruction": "Classify the emotion of each text. Use exactly one of these labels: {labels}. Reply with the label word only.",
    },
    "dbpedia": {
        "labels": ["company", "school", "artist", "athlete", "politician", "transport", "building",
                   "nature", "village", "animal", "plant", "album", "film", "book"],
        "item": "Text",
        "instruction": "Classify the topic of each text. Use exactly one of these labels: {labels}. Reply with the label word only.",
    },
    "banking77": {                                   # 77 intents; the label names are read from the data (labels.json)
        "labels": None,
        "item": "Query",
        "instruction": "Classify the intent of each customer query. Use exactly one of these labels: {labels}. Reply with the label only, written exactly as listed.",
    },
}

TREC_CODES = {"ABBR": "abbreviation", "ENTY": "entity", "DESC": "description",
              "HUM": "human", "LOC": "location", "NUM": "number"}
DBPEDIA_ORDER = ["company", "school", "artist", "athlete", "politician", "transport", "building",
                 "nature", "village", "animal", "plant", "album", "film", "book"]   # label ids 0 to 13


def clean_label(x):
    """Banking77 names such as 'Refund_not_showing_up' or 'reverted_card_payment?' -> lowercase, underscores only."""
    return re.sub(r"[^a-z0-9_]", "", str(x).strip().lower().replace(" ", "_"))


def labels_file(name):
    """labels.json written by prepare_features (needed for datasets whose label names come from the data)."""
    for p in [Path("data") / name / "labels.json"] + sorted(Path("data").glob(f"{name}__*/labels.json")):
        if p.exists():
            return p
    return None


def spec(name):
    s = dict(SPECS[name])
    if s["labels"] is None:
        f = labels_file(name)
        if f is None:
            raise FileNotFoundError(f"run  python -m src.pipeline features --dataset {name}  first (labels.json missing)")
        s["labels"] = json.loads(f.read_text())
    s["instruction"] = s["instruction"].format(labels=", ".join(s["labels"]))
    return s


def _frame(ds_split, text_fn, label_fn):
    df = ds_split.to_pandas()
    return pd.DataFrame({"text": [text_fn(r) for r in df.itertuples()],
                         "label": [label_fn(r) for r in df.itertuples()]})


def load(name):
    from datasets import load_dataset
    if name == "trec":
        errors = []
        try:   # 1) parquet copy of the original CogComp/trec (works with new datasets versions)
            ds = load_dataset("CogComp/trec", revision="refs/convert/parquet")
            names = ds["train"].features["coarse_label"].names
            f = lambda split: _frame(ds[split], lambda r: r.text, lambda r: TREC_CODES[names[r.coarse_label]])
            return f("train"), f("test")
        except Exception as e:
            errors.append(f"CogComp/trec parquet: {e}")
        try:   # 2) SetFit copy
            ds = load_dataset("SetFit/TREC-QC")
            def lab(r):
                code = str(r.label_coarse_original).split(":")[0].upper()
                return TREC_CODES[code]
            f = lambda split: _frame(ds[split], lambda r: r.text, lab)
            return f("train"), f("test")
        except Exception as e:
            errors.append(f"SetFit/TREC-QC: {e}")
        raise RuntimeError("Could not load TREC. Tried:\n  " + "\n  ".join(errors))
    if name == "emotion":
        ds = load_dataset("dair-ai/emotion", "split")
        names = ds["train"].features["label"].names               # sadness, joy, love, anger, fear, surprise
        f = lambda split: _frame(ds[split], lambda r: r.text, lambda r: names[r.label])
        return f("train"), f("test")
    if name == "dbpedia":
        ds = load_dataset("fancyzhx/dbpedia_14")
        text = lambda r: f"{r.title}: {str(r.content).strip()[:300]}"   # cut long abstracts to keep prompts cheap
        f = lambda split: _frame(ds[split], text, lambda r: DBPEDIA_ORDER[r.label])
        return f("train"), f("test")
    if name == "banking77":
        errors = []
        for args in [("PolyAI/banking77", {"revision": "refs/convert/parquet"}), ("PolyAI/banking77", {}), ("banking77", {})]:
            try:
                ds = load_dataset(args[0], **args[1])
                names = [clean_label(n) for n in ds["train"].features["label"].names]
                f = lambda split: _frame(ds[split], lambda r: r.text, lambda r: names[r.label])
                return f("train"), f("test")
            except Exception as e:
                errors.append(f"{args[0]} {args[1]}: {e}")
        raise RuntimeError("Could not load Banking77. Tried:\n  " + "\n  ".join(errors))
    raise ValueError(f"Unknown dataset {name}. Choose from {list(SPECS)}")


def stratified_sample(df, n, seed):
    """Exactly n rows with the label proportions kept as closely as possible (largest remainders).
    Returns df unchanged when it is already small enough."""
    if n is None or len(df) <= n:
        return df.reset_index(drop=True)
    counts = df.label.value_counts()
    exact = counts * n / len(df); base = exact.astype(int)
    extra = (exact - base).sort_values(ascending=False).index[: n - int(base.sum())]
    base[extra] += 1
    out = pd.concat([g.sample(int(base[lab]), random_state=seed) for lab, g in df.groupby("label") if base[lab] > 0])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)
