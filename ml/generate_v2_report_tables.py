import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "experiments" / "results"

REPORT_PATH = RESULTS_DIR / "disease_v2_classification_report.json"
CM_PATH = RESULTS_DIR / "disease_v2_confusion_matrix.csv"


# ============================================================
# 1. CLASS PERFORMANCE TABLE
# ============================================================

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    report = json.load(f)

classes = [
    "Healthy",
    "Maize Blight",
    "Maize Common Rust",
    "Maize Gray Leaf Spot",
    "Rice Bacterial Leaf Blight",
    "Rice Leaf Blast",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
]

rows = []

for disease in classes:
    if disease in report:
        rows.append({
            "Disease": disease,
            "Precision": round(report[disease]["precision"] * 100, 2),
            "Recall": round(report[disease]["recall"] * 100, 2),
            "F1 Score": round(report[disease]["f1-score"] * 100, 2),
            "Support": int(report[disease]["support"]),
        })

performance_df = pd.DataFrame(rows)

performance_path = RESULTS_DIR / "disease_v2_class_performance.csv"
performance_df.to_csv(performance_path, index=False)


# ============================================================
# 2. MAJOR CONFUSION PAIRS
# ============================================================

cm = pd.read_csv(CM_PATH, index_col=0)

confusions = []

for actual in cm.index:
    for predicted in cm.columns:
        if actual != predicted:
            count = int(cm.loc[actual, predicted])

            if count > 0:
                confusions.append({
                    "Actual Disease": actual,
                    "Predicted Disease": predicted,
                    "Errors": count,
                })

confusion_df = pd.DataFrame(confusions)

confusion_df = confusion_df.sort_values(
    "Errors",
    ascending=False
).reset_index(drop=True)

major_confusion_df = confusion_df.head(15)

confusion_path = RESULTS_DIR / "disease_v2_major_confusions.csv"
major_confusion_df.to_csv(confusion_path, index=False)


# ============================================================
# 3. PRINT RESULTS
# ============================================================

print()
print("=" * 75)
print("15-CLASS DISEASE MODEL — CLASS PERFORMANCE")
print("=" * 75)

print(performance_df.to_string(index=False))

print()
print("=" * 75)
print("TOP 15 CONFUSION PAIRS")
print("=" * 75)

print(major_confusion_df.to_string(index=False))

print()
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(f"Class performance:")
print(f"  {performance_path}")

print()
print(f"Major confusion pairs:")
print(f"  {confusion_path}")

print()
print("Report tables generated successfully.")