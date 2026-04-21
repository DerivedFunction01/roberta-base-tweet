#!/usr/bin/env python3
"""
Evaluate a tweet sentiment token-classification model on
`cardiffnlp/tweet_eval` sentiment as if it were a sequence classifier.

This script also scores a reference CardiffNLP sequence-classification model
on the exact same examples so the two runs can be compared side by side.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, project_root)

from tweet.defaults import (  # noqa: E402
    NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
    NORMALIZE_UNICODE_ESCAPES,
    STRIP_QUOTE_ARTIFACTS,
)
from tweet.preprocess import clean_tweet_text  # noqa: E402


MODEL_CHECKPOINT = "token-classification/twitter-roberta-base-sentiment"
TOKENIZER_MODEL = "roberta-base"
BASELINE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_LABELS = ("neg", "neu", "pos")
DATASET_NAME = "cardiffnlp/tweet_eval"
DATASET_SUBSET = "sentiment"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a token-classification tweet sentiment model on cardiffnlp/tweet_eval."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_CHECKPOINT,
        help="Token-classification model checkpoint or local path to evaluate.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=TOKENIZER_MODEL,
        help="Tokenizer checkpoint or local path. Defaults to roberta-base.",
    )
    parser.add_argument(
        "--baseline-model-name",
        type=str,
        default=BASELINE_MODEL_NAME,
        help="Reference sequence-classification model to compare against.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenizer length.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(project_root) / "results" / "tweet_eval_sequence_eval",
        help="Directory for metrics and mismatch outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of test examples to evaluate.",
    )
    parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="Disable the same tweet cleanup used during training.",
    )
    return parser.parse_args()


def _load_sentiment_split(split_name: str):
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)
    if split_name not in dataset:
        available = ", ".join(sorted(dataset.keys()))
        raise KeyError(f"Split '{split_name}' not found. Available splits: {available}")
    return dataset[split_name]


def _sentiment_from_token_label(label_name: str, idx: int | None = None) -> str:
    label = label_name.strip().lower()
    if label.startswith("b-") or label.startswith("i-"):
        label = label[2:]
    if label in {"o", "neu", "neutral"}:
        return "neu"
    if label in {"neg", "negative"}:
        return "neg"
    if label in {"pos", "positive"}:
        return "pos"
    if label.startswith("label_") and idx is not None:
        if idx == 0:
            return "neu"
        if idx in {1, 2}:
            return "neg"
        if idx in {3, 4}:
            return "pos"
    raise ValueError(f"Unsupported token label: {label_name}")


def _sentiment_from_sequence_label(label_name: str, idx: int | None = None) -> str:
    label = label_name.strip().lower()
    if label in {"neg", "negative"}:
        return "neg"
    if label in {"neu", "neutral"}:
        return "neu"
    if label in {"pos", "positive"}:
        return "pos"
    if label.startswith("label_") and idx is not None:
        if idx == 0:
            return "neg"
        if idx == 1:
            return "neu"
        if idx == 2:
            return "pos"
    raise ValueError(f"Unsupported sequence label: {label_name}")


def _dataset_label_to_sentiment(example: dict[str, Any], dataset) -> str:
    value = example["label"]
    feature = dataset.features["label"]
    if hasattr(feature, "int2str") and isinstance(value, int):
        raw_label = feature.int2str(value)
    elif hasattr(feature, "names") and isinstance(value, int):
        raw_label = feature.names[value]
    else:
        raw_label = str(value)
    return _sentiment_from_sequence_label(raw_label)


def _clean_texts(texts: list[str], *, enabled: bool) -> list[str]:
    if not enabled:
        return texts
    return [
        clean_tweet_text(
            text,
            strip_quotes=STRIP_QUOTE_ARTIFACTS,
            normalize_escapes=NORMALIZE_UNICODE_ESCAPES,
            lowercase_dictionary_caps=NORMALIZE_ALL_CAPS_DICTIONARY_WORDS,
        )
        for text in texts
    ]


def _batched(items: list[str], batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _id2label_map(model) -> dict[int, str]:
    id2label_raw = getattr(model.config, "id2label", {})
    return {int(idx): str(label) for idx, label in id2label_raw.items()}


def _predict_token_model_sentiments(
    texts: list[str],
    *,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    id2label = _id2label_map(model)
    total_batches = (len(texts) + batch_size - 1) // batch_size
    predictions: list[dict[str, Any]] = []

    for batch_texts in tqdm(_batched(texts, batch_size), total=total_batches, desc="Scoring token model"):
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = encoded.pop("special_tokens_mask").bool()
        attention_mask = encoded["attention_mask"].bool()
        keep_mask = attention_mask & ~special_tokens_mask
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        keep_mask_np = keep_mask.detach().cpu().numpy()

        for row_probs, row_keep in zip(probabilities, keep_mask_np):
            token_count = int(row_keep.sum())
            if token_count == 0:
                predictions.append(
                    {
                        "predicted": "neu",
                        "scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0},
                        "token_count": 0,
                    }
                )
                continue

            score_sums = {label: 0.0 for label in SENTIMENT_LABELS}
            for token_probs, keep in zip(row_probs, row_keep):
                if not keep:
                    continue
                for idx, score in enumerate(token_probs):
                    label_name = id2label.get(idx, f"label_{idx}")
                    sentiment = _sentiment_from_token_label(label_name, idx)
                    score_sums[sentiment] += float(score)

            averaged_scores = {label: score / token_count for label, score in score_sums.items()}
            predicted = max(SENTIMENT_LABELS, key=lambda label: averaged_scores[label])
            predictions.append(
                {
                    "predicted": predicted,
                    "scores": averaged_scores,
                    "token_count": token_count,
                }
            )

    return predictions


def _predict_sequence_model_sentiments(
    texts: list[str],
    *,
    model,
    tokenizer,
    batch_size: int,
    max_length: int,
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    id2label = _id2label_map(model)
    total_batches = (len(texts) + batch_size - 1) // batch_size
    predictions: list[dict[str, Any]] = []

    for batch_texts in tqdm(_batched(texts, batch_size), total=total_batches, desc="Scoring baseline"):
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.inference_mode():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        for row_probs in probabilities:
            scores: dict[str, float] = {}
            ranked = sorted(
                ((idx, float(score)) for idx, score in enumerate(row_probs)),
                key=lambda item: item[1],
                reverse=True,
            )
            for idx, score in ranked:
                label_name = id2label.get(idx, f"label_{idx}")
                sentiment = _sentiment_from_sequence_label(label_name, idx)
                if sentiment in scores:
                    continue
                scores[sentiment] = score
            for label in SENTIMENT_LABELS:
                scores.setdefault(label, 0.0)
            predicted = max(SENTIMENT_LABELS, key=lambda label: scores[label])
            predictions.append({"predicted": predicted, "scores": scores})

    return predictions


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    label_to_index = {label: index for index, label in enumerate(SENTIMENT_LABELS)}
    confusion = np.zeros((len(SENTIMENT_LABELS), len(SENTIMENT_LABELS)), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        confusion[label_to_index[true_label], label_to_index[pred_label]] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = float(correct / total) if total else 0.0

    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in SENTIMENT_LABELS:
        idx = label_to_index[label]
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        support = int(confusion[idx, :].sum())
        predicted_total = int(confusion[:, idx].sum())

        precision = float(tp / predicted_total) if predicted_total else 0.0
        recall = float(tp / support) if support else 0.0
        f1 = float((2 * precision * recall / (precision + recall))) if (precision + recall) else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precisions)) if precisions else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "total": total,
        "correct": correct,
    }


def _compute_delta_metrics(primary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy_delta": float(primary["accuracy"] - baseline["accuracy"]),
        "macro_precision_delta": float(primary["macro_precision"] - baseline["macro_precision"]),
        "macro_recall_delta": float(primary["macro_recall"] - baseline["macro_recall"]),
        "macro_f1_delta": float(primary["macro_f1"] - baseline["macro_f1"]),
    }


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _format_pct(value: float) -> str:
    return f"{value:.1%}"


def main() -> None:
    args = _parse_args()

    print("# Tweet Eval Sentiment Evaluation")
    print()
    print("## Setup")
    print(f"- Token model: `{args.model_name}`")
    print(f"- Baseline: `{args.baseline_model_name}`")
    print(f"- Tokenizer: `{args.tokenizer_name}`")
    token_model = AutoModelForTokenClassification.from_pretrained(args.model_name)
    baseline_model = AutoModelForSequenceClassification.from_pretrained(args.baseline_model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    print()
    print("## Dataset")
    split = _load_sentiment_split(args.split)
    print(f"- Split: `{args.split}`")
    print(f"- Examples: `{len(split)}`")

    limit = min(args.limit, len(split)) if args.limit is not None else len(split)
    if limit != len(split):
        print(f"- Limit: `{limit}`")

    sample = split.select(range(limit))
    raw_texts = [str(example["text"]) for example in sample]
    texts = _clean_texts(raw_texts, enabled=not args.no_clean_text)

    print()
    print("## Inference")
    token_predictions = _predict_token_model_sentiments(
        texts,
        model=token_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    baseline_predictions = _predict_sequence_model_sentiments(
        texts,
        model=baseline_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print()
    print("## Scoring")
    y_true: list[str] = []
    token_pred_labels: list[str] = []
    baseline_pred_labels: list[str] = []
    token_mismatches: list[dict[str, Any]] = []
    baseline_mismatches: list[dict[str, Any]] = []

    for example, text, token_prediction, baseline_prediction in tqdm(
        zip(sample, texts, token_predictions, baseline_predictions),
        total=limit,
        desc="Collecting metrics",
    ):
        true_label = _dataset_label_to_sentiment(example, sample)
        token_predicted_label = str(token_prediction["predicted"])
        baseline_predicted_label = str(baseline_prediction["predicted"])

        y_true.append(true_label)
        token_pred_labels.append(token_predicted_label)
        baseline_pred_labels.append(baseline_predicted_label)

        if true_label != token_predicted_label:
            token_mismatches.append(
                {
                    "text": text,
                    "text_preview": text[:200],
                    "true_label": true_label,
                    "predicted": token_predicted_label,
                    "scores": token_prediction["scores"],
                    "token_count": int(token_prediction["token_count"]),
                }
            )
        if true_label != baseline_predicted_label:
            baseline_mismatches.append(
                {
                    "text": text,
                    "text_preview": text[:200],
                    "true_label": true_label,
                    "predicted": baseline_predicted_label,
                    "scores": baseline_prediction["scores"],
                }
            )

    token_metrics = _compute_metrics(y_true, token_pred_labels)
    baseline_metrics = _compute_metrics(y_true, baseline_pred_labels)
    deltas = _compute_delta_metrics(token_metrics, baseline_metrics)

    print()
    print("## Overall Results")
    print("| Model | Acc | Macro F1 | Macro P | Macro R |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| Token-classification collapse | {_format_pct(token_metrics['accuracy'])} | "
        f"{_format_pct(token_metrics['macro_f1'])} | {_format_pct(token_metrics['macro_precision'])} | "
        f"{_format_pct(token_metrics['macro_recall'])} |"
    )
    print(
        f"| CardiffNLP baseline | {_format_pct(baseline_metrics['accuracy'])} | "
        f"{_format_pct(baseline_metrics['macro_f1'])} | {_format_pct(baseline_metrics['macro_precision'])} | "
        f"{_format_pct(baseline_metrics['macro_recall'])} |"
    )
    print(
        f"| Delta (token - baseline) | {deltas['accuracy_delta']:+.1%} | {deltas['macro_f1_delta']:+.1%} | "
        f"{deltas['macro_precision_delta']:+.1%} | {deltas['macro_recall_delta']:+.1%} |"
    )

    print()
    print("## Per-Class Breakdown")
    print("| Label | Token F1 | Base F1 | Token P | Base P | Token R | Base R |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label in SENTIMENT_LABELS:
        token_stats = token_metrics["per_class"][label]
        baseline_stats = baseline_metrics["per_class"][label]
        print(
            f"| {label} | {_format_pct(token_stats['f1'])} | {_format_pct(baseline_stats['f1'])} | "
            f"{_format_pct(token_stats['precision'])} | {_format_pct(baseline_stats['precision'])} | "
            f"{_format_pct(token_stats['recall'])} | {_format_pct(baseline_stats['recall'])} |"
        )

    print()
    print("## Confusion Matrices")
    for title, metrics in (
        ("Token-classification collapse", token_metrics),
        ("CardiffNLP baseline", baseline_metrics),
    ):
        print()
        print(f"### {title}")
        print("| True \\ Pred | neg | neu | pos |")
        print("|---|---:|---:|---:|")
        for label, row in zip(SENTIMENT_LABELS, metrics["confusion_matrix"]):
            print(f"| {label} | {int(row[0])} | {int(row[1])} | {int(row[2])} |")

    print()
    print("## Sample Errors")
    print("### Token-classification collapse")
    if token_mismatches:
        for sample_error in token_mismatches[:5]:
            print(f"- `{sample_error['true_label']}` -> `{sample_error['predicted']}`")
            print(f"  - Scores: `{sample_error['scores']}`")
            print(f"  - Preview: {sample_error['text_preview']}...")
    else:
        print("- No mismatches found.")

    print()
    print("### CardiffNLP baseline")
    if baseline_mismatches:
        for sample_error in baseline_mismatches[:5]:
            print(f"- `{sample_error['true_label']}` -> `{sample_error['predicted']}`")
            print(f"  - Scores: `{sample_error['scores']}`")
            print(f"  - Preview: {sample_error['text_preview']}...")
    else:
        print("- No mismatches found.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_output = args.results_dir / "results.json"
    token_mismatch_output = args.results_dir / "token_mismatches.jsonl"
    baseline_mismatch_output = args.results_dir / "baseline_mismatches.jsonl"

    _save_json(
        results_output,
        {
            "dataset": {
                "name": DATASET_NAME,
                "subset": DATASET_SUBSET,
                "split": args.split,
                "limit": limit,
                "clean_text": not args.no_clean_text,
            },
            "token_model_name": args.model_name,
            "baseline_model_name": args.baseline_model_name,
            "tokenizer_name": args.tokenizer_name,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "metrics": {
                "token_classification": token_metrics,
                "baseline": baseline_metrics,
                "delta": deltas,
            },
        },
    )
    with token_mismatch_output.open("w", encoding="utf-8") as f:
        for item in token_mismatches:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with baseline_mismatch_output.open("w", encoding="utf-8") as f:
        for item in baseline_mismatches:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("## Saved Output")
    print(f"- Results: `{results_output}`")
    print(f"- Token-model mismatches: `{token_mismatch_output}`")
    print(f"- Baseline mismatches: `{baseline_mismatch_output}`")


if __name__ == "__main__":
    main()
