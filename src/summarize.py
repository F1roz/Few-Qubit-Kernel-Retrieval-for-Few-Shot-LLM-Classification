"""Cross dataset summary: one table and one figure over every dataset and LLM that has a verdict.
Run:  python -m src.pipeline summary"""
import glob, json, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

rows, verdicts = [], []
for f in sorted(glob.glob("results/*/runs_*/verdict.json")):
    v = json.load(open(f))
    if v.get("encoder", "minilm") != "minilm":
        v["dataset"] = f"{v['dataset']} ({v['encoder']})"
    if v.get("variant"):
        v["dataset"] = f"{v['dataset']} ({v['variant']} test)"
    for sel, acc in v["mean_accuracy"].items():
        rows.append({"dataset": v["dataset"], "llm": v["llm"], "selector": sel, "acc": acc})
    verdicts.append({"dataset": v["dataset"], "llm": v["llm"], "seeds": len(v["seeds"]), **v["verdicts"],
                     "picks differ and g > 1": v["checks"].get("picks differ and g > 1")})
if not rows:
    raise SystemExit("No verdict.json found. Run the analyze stage for at least one dataset first.")
acc = pd.DataFrame(rows).pivot_table(index="selector", columns=["dataset", "llm"], values="acc").round(1)
ver = pd.DataFrame(verdicts)
Path("results").mkdir(exist_ok=True)
acc.to_csv("results/summary_accuracy.csv"); ver.to_csv("results/summary_verdicts.csv", index=False)
print(acc.to_string()); print(); print(ver.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
acc.plot.bar(ax=ax, width=0.8)
ax.set_ylabel("Mean LLM accuracy (%)"); ax.set_xlabel(""); ax.legend(fontsize=7, title="dataset, LLM", title_fontsize=7)
plt.xticks(rotation=30, ha="right", fontsize=7)
Path("figures").mkdir(exist_ok=True)
fig.tight_layout(); fig.savefig("figures/fig7_all_datasets.png", dpi=200); fig.savefig("figures/fig7_all_datasets.pdf")
print("saved results/summary_*.csv and figures/fig7_all_datasets.png")
