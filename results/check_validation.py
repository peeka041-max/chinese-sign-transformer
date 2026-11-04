"""Recompute paired daily20 validation metrics from the published predictions."""

import csv
import math
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def score(rows, classes, column):
    observed = Counter(row["true_word"] for row in rows)
    predicted = Counter(row[column] for row in rows)
    if set(predicted) - classes:
        raise ValueError(f"Unknown prediction in {column}")
    correct = Counter(row["true_word"] for row in rows if row["true_word"] == row[column])
    # 2TP / (true support + predicted support), averaged over the fixed vocabulary.
    macro_f1 = sum(2 * correct[c] / (observed[c] + predicted[c]) for c in classes) / len(classes)
    return sum(correct.values()), macro_f1


def main():
    labels = read_rows(HERE.parent / "realtime_mvp_v2" / "class_order.csv")
    rows = read_rows(HERE / "daily20_validation.csv")
    expected = {(item["gloss_id"], item["display_word"]) for item in labels}
    observed = {(item["gloss_id"], item["true_word"]) for item in rows}
    classes = {word for _, word in expected}
    if len(classes) != 20 or len(rows) != 20 or observed != expected:
        raise ValueError("Expected one validation prediction per daily20 gloss")
    reference = {r["experiment"]: r for r in read_rows(HERE / "initialization_comparison.csv")}
    for name in ("transfer", "scratch"):
        correct, f1 = score(rows, classes, name + "_prediction")
        accuracy = correct / len(rows)
        record = reference[f"daily20_{name}_seed42"]
        if not (math.isclose(accuracy, float(record["val_accuracy"]), abs_tol=1e-7)
                and math.isclose(f1, float(record["val_macro_f1"]), abs_tol=1e-7)):
            raise ValueError(f"{name} predictions disagree with the saved validation metrics")
        print(f"{name:8s} {correct}/{len(rows)}  accuracy={accuracy:.2%}  macro_f1={f1:.4f}")


if __name__ == "__main__":
    main()
