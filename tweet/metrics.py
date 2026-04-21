from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def compute_token_metrics(eval_pred) -> dict[str, float]:
    """Compute token-level accuracy and macro averages, ignoring special tokens."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    y_true: list[int] = []
    y_pred: list[int] = []
    for pred_row, label_row in zip(predictions, labels):
        for pred_id, label_id in zip(pred_row, label_row):
            if int(label_id) == -100:
                continue
            y_true.append(int(label_id))
            y_pred.append(int(pred_id))

    if not y_true:
        return {"accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    accuracy = float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }

