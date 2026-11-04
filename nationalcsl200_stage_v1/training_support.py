"""Shared training/report helpers; no fixed 20-class assumptions."""
import csv
import json
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from protocol import (TRAIN_SIGNERS, VAL_SIGNERS, TEST_SIGNERS, resolve_path,
                      load_manifest_definition, write_csv)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def select_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def classification_metrics(y_true, y_pred, num_classes):
    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class_precision": precision.astype(float).tolist(),
        "per_class_recall": recall.astype(float).tolist(),
        "per_class_f1": f1.astype(float).tolist(),
        "per_class_support": support.astype(int).tolist(),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).astype(int).tolist(),
    }



def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)



def write_evaluation_reports(
    report_dir,
    class_names,
    dataset,
    split_indices,
    split_results,
    summary,
):
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    for split_name in ("val", "test"):
        metrics, y_true, y_pred = split_results[split_name]
        per_class_rows = []
        for class_id, class_name in enumerate(class_names):
            per_class_rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": f"{metrics['per_class_precision'][class_id]:.6f}",
                    "recall": f"{metrics['per_class_recall'][class_id]:.6f}",
                    "f1": f"{metrics['per_class_f1'][class_id]:.6f}",
                    "support": metrics["per_class_support"][class_id],
                }
            )
        write_csv(
            report_dir / f"{split_name}_per_class_metrics.csv",
            ("class_id", "class_name", "precision", "recall", "f1", "support"),
            per_class_rows,
        )

        confusion_path = report_dir / f"{split_name}_confusion_matrix.csv"
        with confusion_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["true\\pred", *class_names])
            for class_name, values in zip(class_names, metrics["confusion_matrix"]):
                writer.writerow([class_name, *values])

        prediction_rows = []
        for dataset_index, true_id, predicted_id in zip(
            split_indices[split_name], y_true, y_pred
        ):
            record = dataset.samples[dataset_index]
            prediction_rows.append(
                {
                    "sample_id": record.path.stem,
                    "signer_id": record.signer_id,
                    "true_class_id": true_id,
                    "true_class_name": class_names[true_id],
                    "predicted_class_id": predicted_id,
                    "predicted_class_name": class_names[predicted_id],
                    "correct": str(true_id == predicted_id).lower(),
                }
            )
        write_csv(
            report_dir / f"{split_name}_predictions.csv",
            (
                "sample_id",
                "signer_id",
                "true_class_id",
                "true_class_name",
                "predicted_class_id",
                "predicted_class_name",
                "correct",
            ),
            prediction_rows,
        )



def compact_metrics(metrics):
    return {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
    }

