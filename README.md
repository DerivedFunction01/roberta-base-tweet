# roberta-base-tweet

Sentiment analysis experiments for tweet data, built around `roberta-base` and `xlm-roberta-base`.

This project should use **token classification**. The idea is to label the whole tweet as `pos`, `neu`, or `neg` by turning each tweet into a token-level training example, so the model learns to separate sentiment boundaries in the same spirit as the language-ID setup.

## Tracks

- `tweet/`: English tweet sentiment baseline with `cardiffnlp/tweet_eval` sentiment
- `tweet-multilingual/`: multilingual tweet sentiment baseline with `cardiffnlp/tweet_sentiment_multilingual`

## Run Shape

- `tweet/train_pipeline.py` builds the tokenized cache folder at the default path from `paths.py`.
- Zip that folder and move it into Colab together with the root `label2id.json`.
- `tweet/train.py` loads the cache from the default path in `paths.py` and trains without any command-line arguments.
- `tweet/defaults.py` holds the assumed default config values for both steps.
- `tweet/mutations.py` provides the light sentence and character augmentation used while building the cache.
- If `label2id.json` is missing, the trainer falls back to `{"neg": 0, "neu": 1, "pos": 2}`.

## Next Game Plan

1. Start with `cardiffnlp/tweet_eval` sentiment, since it is the simplest baseline and the cleanest place to validate the token-labeling approach.
2. Set up the shared manifest/config workflow, similar to the language-ID repo, so runs are reproducible and easy to switch.
3. Build the token-level data loading and preprocessing flow for the English tweet sentiment track.
4. Add tweet cleanup steps before sampling:
   - remove stray quotes and repeated quoting artifacts
   - normalize literal Unicode escape strings
   - optionally lowercase all-uppercase dictionary words with NLTK-assisted checks
   - keep the first pass simple and easy to inspect
5. Build three sentiment pools: `pos`, `neu`, and `neg`.
6. Treat `neu` as a separate pool used for balancing, not as the main target class for token-label mixing.
7. Construct training rows by pairing pools:
   - same-class rows like `pos + pos`
   - mixed rows like `pos + neg`
   - analogous combinations for the other pools
8. Allow reuse with a counter per index so examples can be sampled again if needed, while still limiting overuse.
9. Fine-tune and evaluate `roberta-base` on `cardiffnlp/tweet_eval` with token labels.
10. Build the token-level multilingual tweet sentiment flow.
11. Fine-tune and evaluate `xlm-roberta-base` on `cardiffnlp/tweet_sentiment_multilingual` with token labels.
12. Save each run’s metrics and error samples into per-run result folders.
13. Compare preprocessing choices, label balance, and failure cases across the two tracks.

## What We Want To Measure

- Accuracy
- Macro F1
- Per-class precision and recall
- Confusion matrix
- Token-level span quality, if we keep the whole tweet as a labeled region
- Effects of balancing and reuse counters on the class mix
- A small set of representative misclassifications for manual review

## Working Assumptions

- Keep the English and multilingual tracks separate, but make the config and reporting style consistent.
- Start with a very simple `tweet_eval` sentiment baseline before adding heavier preprocessing.
- Favor simple, reproducible experiments first so we can compare model behavior cleanly.
- Use token classification to make the task look like boundary learning over tweet regions, even though the end goal is tweet-level sentiment.
