"""CLI: uv run gemma-author-probe train"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
from pathlib import Path

from config import CONFIG_PATH, bootstrap_hub_env, load_config
from device import get_device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Frozen LM backbone + probe token authorship classifier (18 classes)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_preview = sub.add_parser("preview", help="Summarize model, head, and parameter counts")
    p_preview.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_preview.add_argument(
        "--config-only",
        action="store_true",
        help="Skip hub download; print config/head summary only",
    )

    p_train = sub.add_parser("train", help="Train probe, head, and optional LoRA on train.csv")
    p_train.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_train.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop after this many optimizer steps (smoke test). Default: config train.max_steps",
    )

    p_run = sub.add_parser("run-test", help="Eval held-out prompts from the saved checkpoint")
    p_run.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_run.add_argument("--checkpoint", type=Path, default=None)

    p_fetch = sub.add_parser(
        "fetch-data",
        help="Download train/test CSVs from Hugging Face (config data.dataset_id)",
    )
    p_fetch.add_argument("--config", type=Path, default=CONFIG_PATH)

    p_export = sub.add_parser(
        "export-data",
        help="Export ntl_open_pool.parquet to data.train_csv (one-off; training reads CSV only)",
    )
    p_export.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_export.add_argument("--parquet", type=Path, default=None)
    p_export.add_argument("--csv", type=Path, default=None)

    p_clean = sub.add_parser(
        "clean-cache",
        help="Free disk: failed hub downloads, xet scratch, pycache; optional uv package cache",
    )
    p_clean.add_argument("--config", type=Path, default=CONFIG_PATH)
    p_clean.add_argument(
        "--no-global-hf",
        action="store_true",
        help="Only clean project hf_cache from config.yaml, not ~/.cache/huggingface",
    )
    p_clean.add_argument(
        "--uv",
        action="store_true",
        help="Also run uv cache clean (~GB; wheels re-download on next uv run)",
    )

    args = parser.parse_args(argv)
    bootstrap_hub_env(args.config)
    cfg = load_config(args.config)

    if args.cmd == "clean-cache":
        from clean_cache import run as run_clean_cache

        run_clean_cache(
            cfg,
            scratch=True,
            global_hf=not args.no_global_hf,
            uv=args.uv,
        )
        return

    if args.cmd == "fetch-data":
        from author_set import ensure_author_set_data

        ensure_author_set_data(cfg)
        print(f"data: ready under {cfg.data.root.resolve()}", flush=True)
        return

    if args.cmd == "export-data":
        from export_ntl_open_pool import DEFAULT_PARQUET_NAME, main as export_main

        parquet = args.parquet or (cfg.data.root / DEFAULT_PARQUET_NAME)
        csv_out = args.csv or (cfg.data.root / cfg.data.train_csv)
        export_main(parquet, csv_out)
        return

    device = get_device(announce=True, prefer=cfg.train.device)

    if args.cmd == "preview":
        from preview import run as run_preview

        run_preview(cfg, device=device, build_from_hub=not args.config_only)
        return

    if args.cmd == "train":
        from train import run

        run(cfg, device=device, max_steps=args.max_steps)
        return

    if args.cmd == "run-test":
        from run_test import run_test

        run_test(cfg, device=device, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
