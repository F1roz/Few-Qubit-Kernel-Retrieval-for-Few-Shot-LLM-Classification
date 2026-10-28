# Design notes, version 5 (what changed and why)

Each change answers a reviewer weakness or comes from our own analysis of the method.

## Own analysis: retrieval only sees RANKINGS
A selector keeps the k highest scores of each query. So any transform of a kernel that keeps the order
of each query's row changes nothing. Three consequences shape the study:
1. The RBF gamma never changes which examples S5 retrieves (it only matters for the geometric difference g).
2. Global depolarizing noise, K -> a K + b with a > 0, never changes the picks. Only LOCAL noise and SHOT
   noise can. So the noise study uses local depolarizing channels plus finite shots.
3. The right way to compare two kernels for retrieval is by how they ORDER the pool, not by their values.
   We add rank agreement (mean Spearman correlation of each query's row under two kernels) next to g,
   which was designed for regression, not ranking.

## Weakness 1: "why expect an advantage at 2 to 5 qubits?"
- Introduction states it openly: at this size every kernel is easy to simulate; the question is whether the
  quantum kernel's notion of similarity (its inductive bias) helps, at the scale where it is being proposed.
- Qubit scaling study (no LLM, free): d = 2 .. 12 with NCA features on all three datasets. Per d: g, rank
  agreement, top-k overlap with RBF, kernel concentration, kernel-target alignment, retrieval AUROC.
  It shows where the IQP kernel starts to order the pool differently from RBF, and where it collapses.

## Weakness 2: simple classical baselines
New selectors (S10 to S15):
- S10 BM25: keyword retrieval (Robertson and Zaragoza 2009), no embedding.
- S11 MMR on cosine of the full embedding (Carbonell and Goldstein 1998), lambda = 0.5 fixed in advance.
- S12 MMR on the IQP kernel: does diversity change the quantum picture?
- S13 trained metric retriever: NCA to 64 dimensions on the full embedding, fitted on the pool labels.
- S14 label oracle: nearest examples that carry the gold label. An UPPER BOUND that uses the answer; it is
  never a competitor, it measures the headroom any selector could gain.
- S15 reciprocal rank fusion (Cormack et al. 2009) of S2 and S7: can the quantum ranking ADD to the
  standard one?
Secondary Holm family (preregistered): S7 vs S2 and S7 vs S13 (the strongest classical opponents).
Exploratory family: S8 vs S5, S12 vs S11, S15 vs S2.

## Weakness 3: scope
- Third LLM: Qwen2.5-7B-Instruct in 4 bits on the T4 (free), d = 5, three datasets.
- Second encoder: all-mpnet-base-v2 (768 dims) on TREC with the open LLM.
- Three seeds on all datasets with the open LLM (free).
- k = 2, 4, 8, 16 on all datasets with the open LLM.
- Noise: local depolarizing noise after every gate layer (density matrices) plus swap-test shots.
- Optional: a small check on real IBM hardware (20 queries x 50 pool items, 4 qubits).

## Weakness 4: a null result needs a strong "why"
- Equivalence test (TOST, two one-sided tests) against the preregistered margin of 2 points: turns
  "no difference found" into "difference shown to be smaller than 2 points".
- Minimum detectable effect at 80% power for every comparison.
- Headroom: oracle minus best real selector. If the oracle itself gains little, no kernel can gain much.
- Difficulty strata: the IQP minus RBF gap on easy, medium and hard queries.

## Weakness 5: tuning by the neighbour vote
- Bandwidth study with the open LLM: choose c by LLM accuracy on the 100 validation items, and report the
  test accuracy of the quantum selectors at EVERY c. If the verdict holds at every c, tuning cannot explain it.

## Fixes from the independent review (before release)
- Difficulty strata now use cosine on the full embedding (S2) as the reference. Using RBF's own retrieval
  AUROC would favour IQP on the "hard" third by regression to the mean.
- g(cosine || IQP) uses cos^2, which is positive semidefinite (max(cos, 0)^2 need not be).
- The headline interval in the paper and the equivalence table now come from the same bootstrap.
- Qwen 7B runs seed 0 only by default (pipeline), as the paper says.
- Every result sentence that draws a conclusion now checks the numbers in both directions
  (src/paper_text.py); nothing in main.tex states an outcome by hand.

## Version 6: answers to the second review
- More qubits WITH the LLM: NCA features with 6, 8, 10, 12 qubits, open LLM, all four datasets
  (`nca --llm llm2`). Answers "a null at 2-5 LDA features is built into the design".
- Trained kernels on equal terms: S16 (IQP) and S17 (ARD-RBF) learn one input scale per feature by centred
  kernel-target alignment on the pool labels (same objective, optimiser, data and parameter count).
  Answers "a fixed quantum kernel is a straw man".
- A harder task: Banking77 (77 intents), where five LDA features cannot separate the classes.
- Statistical power: 2,000-item test sets (`--big`) for Emotion, DBpedia and Banking77 with the open LLM.
- The margin of 2 points is justified in the paper and checked against seed and order variation by the code.
- g is reported at the median gamma too (the minimum favours "no difference").
- Every analysis outside the preregistered families is labelled exploratory.
- Parser: exact label, then the longest label the reply starts with, then a unique prefix (77 intent names);
  max new tokens 24. Label probabilities are computed in small batches and only for <= 20 labels.
