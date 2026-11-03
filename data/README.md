Training reads **ntl_open_pool.csv** (`data.train_csv`). Evaluation uses **test.csv**; `data.test_fraction` picks that share of complete `prompt_id`s (`data.seed`).

```bash
# ntl_open_pool.csv and test.csv in data/
uv run gemma-author-probe train
```

To regenerate the CSV from parquet: `uv run gemma-author-probe export-data` (defaults to `data/ntl_open_pool.parquet`).

Parquet columns map to CSV as: `id` → `id`, `prompt_group` → `prompt_id`, `text` → `text`, `model` → `model`.

Training batches use **ntl_open_pool.csv**. With `data.require_all_models: true` (default), only prompts that contain all 18 models are batched. With `false`, any prompt with at least one model is eligible; if a sampled model id is missing on that prompt, a random model present on that prompt is used instead. A repeated model on the same prompt keeps the first row. Test eval still uses complete prompts (18 models each) from **test.csv** per `data.test_fraction`.
