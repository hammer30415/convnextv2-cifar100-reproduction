from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"D:\ConvNeXtV2-course-project")
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

v1 = pd.read_csv(ROOT / "runs/formal-v1-atto/history.csv")
v2 = pd.read_csv(ROOT / "runs/formal-v2-atto/history.csv")
nogrn = pd.read_csv(ROOT / "runs/ablation-v2-no-grn/history.csv")

plt.figure(figsize=(8, 5))
plt.plot(v1["epoch"], v1["val_accuracy"] * 100, label="ConvNeXt V1-Atto")
plt.plot(v2["epoch"], v2["val_accuracy"] * 100, label="ConvNeXt V2-Atto")
plt.plot(nogrn["epoch"], nogrn["val_accuracy"] * 100, label="V2 without GRN")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "validation_accuracy_curves.png", dpi=300)
plt.close()

models = ["V1-Atto", "V2-Atto", "V2 without GRN"]
test_accuracy = [79.47, 79.81, 74.84]

plt.figure(figsize=(7, 5))
bars = plt.bar(models, test_accuracy)
plt.ylabel("Test Accuracy (%)")
plt.ylim(70, 85)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, test_accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.2,
        f"{value:.2f}%",
        ha="center",
    )

plt.tight_layout()
plt.savefig(FIG / "test_accuracy_comparison.png", dpi=300)
plt.close()

print("图表已保存到:", FIG)
