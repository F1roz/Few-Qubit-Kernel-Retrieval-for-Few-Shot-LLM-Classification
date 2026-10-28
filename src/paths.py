"""Where every file lives. One folder per dataset, and one runs folder per LLM.

The dataset comes from configs/config.yaml (data.dataset) unless the environment variable
QK_DATASET is set; the LLM profile (llm, llm2, llm3) comes from QK_LLM; the sentence encoder
(minilm or mpnet) comes from QK_EMB, and QK_VARIANT=big selects the larger test set. src.pipeline sets them.
The main encoder keeps the plain folder names (data/trec); another encoder gets its own (data/trec__mpnet)."""
import os
from pathlib import Path
import yaml

CFG = yaml.safe_load(open("configs/config.yaml"))
DATASET = os.environ.get("QK_DATASET", CFG["data"]["dataset"])
PROFILE = os.environ.get("QK_LLM", "llm")
EMB = os.environ.get("QK_EMB", CFG["data"].get("main_encoder", "minilm"))
EMB_MODEL = CFG["data"]["encoders"][EMB]
TAG = DATASET if EMB == CFG["data"].get("main_encoder", "minilm") else f"{DATASET}__{EMB}"
VARIANT = os.environ.get("QK_VARIANT", "")            # "big": a larger test set (src.pipeline --big)
if VARIANT:
    TAG = f"{TAG}__{VARIANT}"

DATA = Path("data") / TAG                   # train.csv, test.csv, splits_seed*.csv
FEAT = DATA / "features"                    # embeddings and reduced features (.npy)
RES = Path("results") / TAG             # analyses before the LLM, tuned settings
RUNS = RES / f"runs_{PROFILE}"              # LLM runs and result tables for this LLM
FIG = Path("figures") / f"{TAG}_{PROFILE}"
TUNED = RES / "tuned.json"

for p in (FEAT, RUNS, FIG):
    p.mkdir(parents=True, exist_ok=True)


def llm_settings():
    s = dict(CFG[PROFILE])
    out = {k: s[k] for k in ["provider", "model", "temperature", "max_tokens", "cache_file"]}
    out["load_in_4bit"] = bool(s.get("load_in_4bit", False))
    return out
