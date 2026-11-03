"""Download train/test CSVs from the Hugging Face author-set dataset."""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from config import Config
from env import hf_hub_token

DEFAULT_AUTHOR_SET_DATASET = "saquiboye/author-set"


def required_data_files(cfg: Config) -> tuple[str, ...]:
    names = [cfg.data.train_csv]
    if cfg.data.test_fraction > 0.0:
        names.append(cfg.data.test_csv)
    return tuple(names)


def _data_files_ready(root: Path, filenames: tuple[str, ...]) -> bool:
    return bool(filenames) and all((root / name).is_file() for name in filenames)


def _clean_data_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _download_files(cfg: Config, filenames: tuple[str, ...]) -> None:
    root = cfg.data.root
    dataset = cfg.data.dataset_id
    token = hf_hub_token()
    for filename in filenames:
        dest = root / filename
        print(f"data: downloading hf://datasets/{dataset}/{filename}", flush=True)
        hf_hub_download(
            repo_id=dataset,
            filename=filename,
            repo_type="dataset",
            local_dir=str(root),
            token=token,
        )
        if not dest.is_file():
            raise FileNotFoundError(f"download finished but {dest} is missing")


def ensure_author_set_data(cfg: Config) -> None:
    """Ensure train/test CSVs exist; wipe and re-fetch from Hub if the set is incomplete."""
    root = cfg.data.root
    required = required_data_files(cfg)
    if _data_files_ready(root, required):
        return
    if root.is_dir() and any(root.iterdir()):
        print(
            f"data: missing required file(s) under {root}; cleaning folder before re-download",
            flush=True,
        )
        _clean_data_dir(root)
    else:
        root.mkdir(parents=True, exist_ok=True)
    _download_files(cfg, required)
