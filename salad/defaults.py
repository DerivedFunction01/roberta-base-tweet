from __future__ import annotations

from paths import path


DATASET_NAME = "OpenSafetyLab/Salad-Data"
SUBSET = "base_set"
TEXT_COLUMN = "question"
LABEL_COLUMN = "1-category"
MAX_SENTENCES = 3
MIN_LATIN_RATIO = 0.5
CACHE_DIR = path("salad", "salad_cache_dir")
NEUTRAL_DATASET_NAME = "teknium/OpenHermes-2.5"
NEUTRAL_SPLIT = "train"
NEUTRAL_TEXT_COLUMN = "conversations"
NEUTRAL_SAMPLE_FRACTION = 0.05
NEUTRAL_MAX_SENTENCES = 3
NEUTRAL_MIN_LATIN_RATIO = 0.5
NEUTRAL_CACHE_DIR = path("salad", "salad_outside_cache_dir")
