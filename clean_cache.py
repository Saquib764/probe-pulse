"""Safe cache cleanup: runtime device memory and on-disk scratch only."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from config import Config, project_root
from device import release_device_memory

DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"


def _remove_tree(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        size = path.stat().st_size
        path.unlink()
        return size
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    shutil.rmtree(path)
    return size


def _clean_hf_hub_scratch(hub: Path, *, label: str) -> tuple[int, list[str]]:
    freed = 0
    lines: list[str] = []
    if not hub.is_dir():
        return freed, lines

    locks = hub / ".locks"
    if locks.is_dir():
        for lock in locks.iterdir():
            if lock.is_file():
                nbytes = _remove_tree(lock)
                freed += nbytes
                lines.append(f"removed {label} hub lock {lock.name}")

    for incomplete in hub.rglob("*.incomplete"):
        nbytes = _remove_tree(incomplete)
        freed += nbytes
        try:
            rel = incomplete.relative_to(hub)
        except ValueError:
            rel = incomplete.name
        lines.append(f"removed {label} incomplete {rel} ({nbytes // (1024 * 1024)} MiB)")

    return freed, lines


def _clean_hf_xet(hf_home: Path, *, label: str) -> tuple[int, list[str]]:
    xet = hf_home / "xet"
    if not xet.is_dir():
        return 0, []
    nbytes = _remove_tree(xet)
    return nbytes, [f"removed {label} xet ({nbytes // 1024} KiB)"]


def clean_scratch(cfg: Config, *, global_hf: bool = True) -> list[str]:
    """Drop bytecode caches, stale hub locks, HF xet scratch, and failed downloads. Keeps model blobs."""
    root = project_root()
    freed = 0
    lines: list[str] = []

    for cache_dir in root.rglob("__pycache__"):
        if ".venv" in cache_dir.parts:
            continue
        nbytes = _remove_tree(cache_dir)
        freed += nbytes
        lines.append(f"removed {cache_dir.relative_to(root)} ({nbytes // 1024} KiB)")

    for label, hf_home in (
        ("project", cfg.hf_cache),
        ("global", DEFAULT_HF_HOME),
    ):
        if label == "global" and not global_hf:
            continue
        hub = hf_home / "hub"
        chunk, part = _clean_hf_hub_scratch(hub, label=label)
        freed += chunk
        lines.extend(part)
        chunk, part = _clean_hf_xet(hf_home, label=label)
        freed += chunk
        lines.extend(part)

    if not lines:
        lines.append("nothing to remove (scratch already clean)")
    else:
        lines.append(f"scratch freed ~{freed // (1024 * 1024)} MiB")
    return lines


def clean_uv_cache() -> list[str]:
    """Drop uv wheel/source cache (safe; packages re-fetch on next uv run)."""
    before = 0
    uv_root = Path.home() / ".cache" / "uv"
    if uv_root.is_dir():
        before = sum(f.stat().st_size for f in uv_root.rglob("*") if f.is_file())
    proc = subprocess.run(
        ["uv", "cache", "clean"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "uv cache clean failed").strip()
        return [f"uv cache: skipped ({msg})"]
    return [f"uv cache: freed ~{before // (1024 * 1024)} MiB"]


def run(cfg: Config, *, scratch: bool, global_hf: bool = True, uv: bool = False) -> None:
    release_device_memory()
    print("clean-cache: released device allocator cache in this process", flush=True)
    if scratch:
        for line in clean_scratch(cfg, global_hf=global_hf):
            print(f"clean-cache: {line}", flush=True)
    if uv:
        for line in clean_uv_cache():
            print(f"clean-cache: {line}", flush=True)
