# gemma-author-probe

Authorship classifier on a frozen causal LM. Learnable probe tokens are appended after the passage, tapped at selected layers, and fed to a linear head plus a supervised contrastive head. Optional LoRA adapters cover disjoint layer ranges.

The package name on the CLI is `gemma-author-probe`. Python 3.11+. Dependencies are managed with [uv](https://docs.astral.sh/uv/). Config is `config.yaml`.

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and set `HF_TOKEN`. `google/gemma-3-270m` is gated; accept the license on Hugging Face first. The token also raises Hub rate limits for other models. This package's `.env` wins. Missing keys are filled from `.env` in sibling folders next to this repo: `decision_gemma`, `modern_bert_image_gen`, `florence_trainer`, and `sana_trainer`.

`data/` is gitignored. CSVs come from [saquiboye/author-set](https://huggingface.co/datasets/saquiboye/author-set) on first train, or from the fetch command. Set `data.dataset_id`, `data.train_csv`, and `data.test_csv` in `config.yaml`.

## Commands

```bash
uv run gemma-author-probe fetch-data
uv run gemma-author-probe preview
uv run gemma-author-probe preview --config-only
uv run gemma-author-probe train
uv run gemma-author-probe train --max-steps 2
uv run gemma-author-probe run-test
uv run gemma-author-probe clean-cache
```

`export-data` writes `ntl_open_pool.parquet` out to the train CSV. Training itself reads CSV only.

`train.device` is `auto`, `cpu`, `mps`, or `cuda`. On macOS stay on torch 2.6.x; 2.15 breaks MPS linear. `PYTORCH_ENABLE_MPS_FALLBACK` is set by the CLI.

## Model

Frozen causal LM with max length 512. Set `model_id`, `tap_layers`, and `lora.*.layers` in `config.yaml` (for example `google/gemma-3-270m` with 18 blocks, or `Qwen/Qwen3-0.6B-Base` with 28). Each row is tokenized as the passage text only. Tokenization leaves room for `probe.probe_count` learnable probe tokens (default 10), each initialized from the mean embedding of tokens in a structure-focused question, appended in fixed order after the passage. At each tapped layer, all probe hidden states are concatenated; tap vectors are concatenated and fed to the linear head. Supervised contrastive loss uses the `contrast` block (two-layer head on the pooled vector, then L2 normalize). Token embeddings are always frozen. Up to four named LoRA adapters (`lora.lora_1` through `lora_4`) target disjoint 1-indexed layer ranges; each has `enabled`, `layers`, `r`, `alpha`, `dropout`, and `lr`. Probes, classifier head, and contrast head are configured under `probe`, `head`, and `contrast`. Each optimizer step samples `train.prompt_count` prompt ids and `train.batch_size / train.prompt_count` model ids, and uses only those (prompt, author) rows.

Qwen3-0.6B-Base and Qwen3-1.7B share a 28-block tap map. 4B and 8B have 36 blocks, 14B has 40, 32B has 64. Retarget `tap_layers` and `lora.*.layers` for those.
