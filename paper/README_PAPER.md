# The paper: how to finish it

The paper is not tied to any conference. One line in `configs/config.yaml` picks the template:
`paper: venue: ieee | springer | acm | arxiv` (IEEE conferences, Springer LNCS, ACM sigconf, or a plain
two-column layout for arXiv). Overleaf has all four templates.
Everything in the paper that depends on results is written by the code. Your job is to run, read and check.

## What you fill by hand (5 minutes, once)
Open `configs/config.yaml`, section `paper:`
- `venue` (the template), `authors`, `emails`, `department`, `institution`, `city`, `country`
- `funding` (optional sentence)
- `repo_url`: make a public GitHub repository, upload this project (without `.env` or keys), paste its link

The main LLM is already set: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`). The access dates and the number
of calls are read from the LLM cache, so the paper writes them itself.

## What the code fills (one command)
`python -m src.pipeline paper` (notebook Step 12b) writes:
| File | Content |
|---|---|
| `results_macros.tex` | every number in the text |
| `results_text.tex` | every result sentence (RQ1 answer, trends, interpretation, ablation, shots, errors, discussion, conclusion), written by fixed rules in `src/paper_text.py` from your numbers |
| `venue.tex`, `authors.tex` | the template and the author block, from the config |
| `tables/*.tex` | every results table |
| `figures/*.pdf, *.png` | every plot (replaces the PREVIEW plots) |
Then `cd paper && python word/build_docx.py` rebuilds `Paper.docx` with the same content.

## What you check before submitting
1. Compile `main.tex` and `supplement.tex` on Overleaf (upload the `paper` folder). No red `[...]`, no `XYZ`, no PREVIEW stamp.
2. Read every sentence in `results_text.tex` against the tables. The rules only state what the numbers show;
   if one reads wrong for your data, change the rule in `src/paper_text.py` (or the sentence) and rerun.
3. Length: the main paper is about 14 pages plus a 4-page supplement (IEEE or arXiv template), which suits
   journals and venues without a strict limit. The geometry, difficulty, calibration and ablation material is
   already in `supplement.tex`. For an 8-page conference, move next: Fig. `fig_scaling`, Fig. `fig_noise`,
   Fig. `fig_bandwidth`, Fig. `fig_ksweep` and Table `tab_robust` (cut and paste each whole `figure`/`table` block
   into `supplement.tex` and replace the reference in the text by "supplementary material").
4. Open every reference once (authors, year, venue). The 2026 arXiv paper (cajas2026medical) lists only its
   first author; add the others from its arXiv page.
5. Check the venue's rules on AI use and edit `paper: ai_disclosure:` in the config to match.
6. If you did not run the optional IBM hardware check, the paper says so in one sentence; that is fine.
