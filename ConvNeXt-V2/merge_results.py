from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\ConvNeXtV2-course-project")

main = pd.read_csv(ROOT / "results/main_results.csv")
eff = pd.read_csv(ROOT / "results/efficiency_results.csv")

summary = main.merge(eff, on="model", how="left")
summary.to_csv(
    ROOT / "results/final_results_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

print(summary.to_string(index=False))
print("汇总表已保存到:", ROOT / "results/final_results_summary.csv")
