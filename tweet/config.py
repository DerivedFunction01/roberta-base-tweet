from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".evaluation_config.json"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"

LABEL_NAMES = ["neg", "neu", "pos"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def default_manifest() -> dict[str, Any]:
    """Create a small manifest with one tweet-eval run."""
    config_id = "tweet_eval_default"
    return {
        "runs": [config_id],
        "configs": [
            {
                "id": config_id,
                "run": "tweet_eval",
                "dataset_name": "cardiffnlp/tweet_eval",
                "subset": "sentiment",
                "model_name": "roberta-base",
                "tokenizer_name": "roberta-base",
                "task_type": "token-classification",
                "results_dir": f"results/{config_id}",
                "max_length": 128,
                "same_class_ratio": 0.5,
                "reuse_limit": 2,
                "train_examples": 12_000,
                "validation_examples": 2_000,
                "test_examples": 2_000,
                "batch_size": 8,
                "epochs": 3,
                "learning_rate": 2e-5,
                "strip_quote_artifacts": True,
                "normalize_unicode_escapes": True,
                "normalize_all_caps_dictionary_words": False,
            }
        ],
    }


def load_or_create_manifest(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the shared manifest, creating a default one if needed."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(default_manifest(), f, ensure_ascii=False, indent=2)
        print(f"Created default config at {config_path}")
        print("Edit the manifest and rerun.")
        raise SystemExit(0)

    with config_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected a JSON object in {config_path}")
    if not isinstance(manifest.get("configs"), list) or not isinstance(manifest.get("runs"), list):
        raise ValueError(f"Expected top-level 'configs' and 'runs' lists in {config_path}")
    return manifest


def select_config(
    manifest: dict[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    config_id: str | None = None,
) -> dict[str, Any]:
    """Select the active config from the manifest."""
    configs = [
        config
        for config in manifest["configs"]
        if isinstance(config, dict) and str(config.get("id", "")).strip() in manifest["runs"]
    ]
    if config_id is None:
        if len(configs) != 1:
            raise ValueError(
                f"Expected exactly one active config in {config_path}, got {[cfg.get('id') for cfg in configs]}"
            )
        selected = configs[0]
    else:
        matches = [config for config in configs if str(config.get("id", "")).strip() == config_id]
        if not matches:
            raise ValueError(f"Config id '{config_id}' is not active in {config_path}")
        selected = matches[0]

    config_id_value = str(selected.get("id", "")).strip()
    if not config_id_value:
        raise ValueError(f"Missing config id in {config_path}")
    merged = dict(default_manifest()["configs"][0])
    merged.update(selected)
    merged["id"] = config_id_value
    merged["run"] = str(merged.get("run", "tweet_eval")).strip().lower()
    return merged


def resolve_path(base_dir: Path, value: Any) -> Path:
    """Resolve a possibly relative path against the project root."""
    candidate = Path(str(value)) if value is not None and str(value).strip() else base_dir
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
