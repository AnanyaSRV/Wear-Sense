
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("data") / "UCI HAR Dataset"
OUTPUT = Path("report") / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv(
    ROOT / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["activity_id", "activity"]
)

y_train = pd.read_csv(
    ROOT / "train" / "y_train.txt",
    header=None
)[0]

counts = y_train.value_counts().sort_index()
names = labels.set_index("activity_id")["activity"]
counts.index = [names.loc[i] for i in counts.index]

plt.figure(figsize=(10, 6))
bars = plt.bar(counts.index, counts.values)

plt.title("WearSense: Training Activity Distribution")
plt.xlabel("Physical Activity")
plt.ylabel("Number of Samples")
plt.xticks(rotation=25, ha="right")

for bar, value in zip(bars, counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    OUTPUT / "activity_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("Activity distribution:")
print(counts)
print(f"\nChart saved to: {OUTPUT / 'activity_distribution.png'}")
