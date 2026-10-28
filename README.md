# Quantum Kernel Example Selection for LLM Few Shot Prompting (version 6)

Code for the whole project. The Hands-On Guide explains every step; `DESIGN_NOTES.md` explains why version 5
looks the way it does (what each new part answers).

## Easiest way: Google Colab
Upload this zip to your Google Drive, open `run_all.ipynb` in Colab (T4 GPU runtime), and run the cells top to bottom.

## What the code covers
- **Four datasets**: `trec` (main), `emotion`, `dbpedia`, and `banking77` (77 intents, the harder task). Pick one with `--dataset`.
  `--big` uses a 2,000-item test set (Emotion, DBpedia, Banking77) in its own folders.
- **Seventeen selectors**: S1 random, S2 cosine (full embedding), S3 cosine (LDA), S4 Euclidean, S5 RBF,
  S6 quantum angle, S7 quantum IQP, S8 projected quantum kernel, S9 IQP with 1024 shots (robustness),
  S10 BM25, S11 MMR on cosine, S12 MMR on IQP, S13 trained metric (NCA-64), S14 label oracle (upper bound),
  S15 rank fusion of cosine and IQP, S16 quantum IQP with trained input scales, S17 RBF with trained input scales
  (both trained the same way: centred kernel-target alignment on the pool labels; `src/trained_kernels.py`).
- **Three LLMs** (`--llm`): `llm` (Claude Haiku 4.5, paid API), `llm2` (Qwen2.5 1.5B, free on the GPU),
  `llm3` (Qwen2.5 7B in 4 bits, free on the GPU).
- **Two encoders** (`--emb`): `minilm` (main) and `mpnet` (robustness).
- **Studies**: ablation, qubit scaling (2 to 12 qubits), gate noise (no LLM); bandwidth study, k sweep, order (open LLM);
  optional check on IBM hardware.
- **Statistics**: preregistered primary, secondary and exploratory families (Holm), cluster bootstrap,
  TOST equivalence test, minimum detectable effect, difficulty strata, headroom to the oracle.

## From a terminal (run inside this folder)
Every command below starts with `python -m src.pipeline`. Loop over the three datasets where it says "each".

| Step | Command | Cost |
|---|---|---|
| setup | `pip install -r requirements.txt`; `check` (all PASS); `python tests/smoke_pipeline.py` (SMOKE TEST PASSED) | free |
| data | `features --dataset <each>`; `features --dataset trec --emb mpnet` | free |
| data | `features --dataset banking77`; `features --dataset <emotion, dbpedia, banking77> --big` | free |
| before LLM | `pre_llm --dataset <each>`; `pre_llm --dataset trec --emb mpnet --seeds 0` | free |
| before LLM | `pre_llm --dataset banking77 --seeds 0`; `pre_llm --dataset <all four> --method nca --seeds 0`; `pre_llm --dataset <x> --big --seeds 0` | free |
| studies | `ablate`, `scaling`, `noise` with `--dataset <each>` | free |
| LLM setup | set `llm.model` in `configs/config.yaml`, export the API key; `estimate --dataset <each>` | free |
| pilot | `zeroshot --dataset trec`, `pilot --dataset trec`; write and commit `PREREGISTRATION.md` (notebook Step 9) | small |
| main LLM | `main --dataset <each> --seeds 0`; `robust --dataset trec`; `main --dataset trec --seeds 1,2` | paid |
| main LLM, extra | `ksweep --dataset trec`; `pre_llm --dataset trec --method nca --seeds 0`, `nca --dataset trec` | paid |
| open LLM | `main --dataset <each> --llm llm2` (3 seeds); `ksweep --dataset <each> --llm llm2` | free on GPU |
| open LLM | `order --dataset trec --llm llm2`; `bwstudy --dataset trec --llm llm2` | free on GPU |
| encoder | `main --dataset trec --emb mpnet --llm llm2 --seeds 0` | free on GPU |
| more qubits | `nca --dataset <all four> --llm llm2` (6, 8, 10, 12 qubits) | free on GPU |
| harder task | `main --dataset banking77 --llm llm2 --seeds 0` (main LLM optional) | free on GPU |
| power | `main --dataset <emotion, dbpedia, banking77> --big --llm llm2 --seeds 0` | free on GPU |
| 7B LLM | `main --dataset <each> --llm llm3 --seeds 0` | free on GPU |
| optional | `hardware --dataset trec --simulate`, then without `--simulate` (needs qiskit and an IBM account) | free tier |
| analysis | `analyze --dataset <each> --llm <llm, llm2, llm3>`; `analyze --dataset banking77 --llm llm2`; `analyze --dataset trec --emb mpnet --llm llm2`; `analyze --dataset <x> --big --llm llm2`; `summary` | free |
| paper | `paper`: fills `paper/tables`, `paper/figures`, `paper/results_macros.tex`, `paper/results_text.tex` | free |

API key: `export ANTHROPIC_API_KEY=...` (never put it in a file you commit). Every LLM answer is cached in
`results/llm_cache.jsonl`. Never delete that file. If any step stops, run the same command again.
`--backend numpy` gives the same kernels faster than PennyLane; use it only after `check` passes.

## Folder map
- `configs/config.yaml`  every setting, including the decision rules, comparison families and all LLM slots
- `src/`  the code (`pipeline.py` runs every stage; `scaling_study.py`, `noise_study.py`, `bandwidth_study.py`,
  `hardware_check.py` are the studies; `paper_tables.py`, `paper_text.py`, `paper_figures.py` fill the paper)
- `tests/`  `test_core.py` (unit tests) and `smoke_pipeline.py` (every stage on fake data)
- `data/<dataset>/`, `results/<dataset>/runs_<llm>/`, `figures/<dataset>_<llm>/`  filled by the scripts
  (the MPNet runs use `data/trec__mpnet/` and `results/trec__mpnet/`)
- `paper/`  the paper: `main.tex`, `supplement.tex`, `refs.bib`, `Paper.docx`; see `paper/README_PAPER.md`
