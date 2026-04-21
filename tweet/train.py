from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

BASE_DIR = Path(".")
TOKENIZED_DATASET_DIR = BASE_DIR / "tokenized_dataset"
STANDALONE_RESULTS_DIR = BASE_DIR / "results" / "tweet_eval_standalone"

MODEL_CHECKPOINT = "roberta-base"
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
EPOCHS = 3.0
LEARNING_RATE = 2e-5
SEED = 42

DEFAULT_LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_label_map() -> dict[str, int]:
    label_map_path = Path("label2id.json")
    if label_map_path.exists():
        with label_map_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object in {label_map_path}")
        label2id = {str(label): int(idx) for label, idx in data.items()}
    else:
        label2id = dict(DEFAULT_LABEL2ID)

    expected_ids = list(range(len(label2id)))
    actual_ids = sorted(label2id.values())
    if actual_ids != expected_ids:
        raise ValueError(f"Label ids must be contiguous starting at 0; got {actual_ids}")
    return label2id


def load_tokenized_cache(data_dir: Path) -> dict[str, Dataset]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Tokenized cache not found: {data_dir}")

    loaded = load_from_disk(str(data_dir))
    if isinstance(loaded, DatasetDict):
        return {name: split for name, split in loaded.items()}
    if isinstance(loaded, Dataset):
        return {"train": loaded}
    raise TypeError(f"Unsupported dataset cache type: {type(loaded)!r}")


def choose_split(splits: dict[str, Dataset], *names: str, required: bool = True) -> Dataset | None:
    for name in names:
        if name in splits:
            return splits[name]
    if required:
        raise KeyError(f"Could not find any of these splits: {names}")
    return None


def make_compute_metrics(id2label: dict[int, str]):
    label_ids = sorted(id2label.keys())

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_ids = np.argmax(predictions, axis=-1)
        mask = labels != -100
        true_ids = labels[mask]
        pred_ids = predicted_ids[mask]

        if true_ids.size == 0:
            return {
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
            }

        accuracy = float(np.mean(true_ids == pred_ids))
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for label_id in label_ids:
            tp = float(np.sum((pred_ids == label_id) & (true_ids == label_id)))
            fp = float(np.sum((pred_ids == label_id) & (true_ids != label_id)))
            fn = float(np.sum((pred_ids != label_id) & (true_ids == label_id)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return {
            "accuracy": accuracy,
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)),
        }

    return compute_metrics


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    label2id = load_label_map()
    id2label = {idx: label for label, idx in label2id.items()}
    splits = load_tokenized_cache(TOKENIZED_DATASET_DIR)

    train_dataset = choose_split(splits, "train")
    validation_dataset = choose_split(splits, "validation", "eval", "dev")
    test_dataset = choose_split(splits, "test", "validation", "eval", "dev", required=False)

    STANDALONE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    training_args = TrainingArguments(
        output_dir=str(STANDALONE_RESULTS_DIR / "checkpoints"),
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        seed=SEED,
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=max(1, mp.cpu_count() // 2),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(id2label),
    )

    print("=" * 80)
    print("TRAINING TWEET SENTIMENT TOKEN CLASSIFIER")
    print("=" * 80)
    print(f"Data dir: {TOKENIZED_DATASET_DIR}")
    print(f"Model: {MODEL_CHECKPOINT}")
    print(f"Labels: {label2id}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(validation_dataset)}")
    if test_dataset is not None:
        print(f"Test size: {len(test_dataset)}")
    print(f"Output: {STANDALONE_RESULTS_DIR}")

    first_example = train_dataset[0]
    tokens = tokenizer.convert_ids_to_tokens(first_example["input_ids"])
    labels = [id2label[label] for label in first_example["labels"] if label != -100]
    print(f"Sample tokens: {tokens}")
    print(f"Sample labels: {labels}")

    trainer.train()
    validation_metrics = trainer.evaluate(validation_dataset)
    test_metrics = None
    if test_dataset is not None:
        test_predictions = trainer.predict(test_dataset)
        test_metrics = make_compute_metrics(id2label)((test_predictions.predictions, test_predictions.label_ids))

    trainer.save_model(str(STANDALONE_RESULTS_DIR / "final_model"))
    tokenizer.save_pretrained(str(STANDALONE_RESULTS_DIR / "final_model"))
    save_json(
        STANDALONE_RESULTS_DIR / "results.json",
        {
            "data_dir": str(TOKENIZED_DATASET_DIR),
            "model_checkpoint": MODEL_CHECKPOINT,
            "label2id": label2id,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "train_size": len(train_dataset),
            "validation_size": len(validation_dataset),
            "test_size": len(test_dataset) if test_dataset is not None else None,
        },
    )

    print("\nSaved:")
    print(f"  {STANDALONE_RESULTS_DIR / 'results.json'}")
    print(f"  {STANDALONE_RESULTS_DIR / 'final_model'}")


if __name__ == "__main__":
    main()
