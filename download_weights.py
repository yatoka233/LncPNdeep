# -*- coding: utf-8 -*-
"""
Download LncPNdeep model weights from Hugging Face Hub.

Weights repo: https://huggingface.co/yatoka/LncPNdeep

Example:
  python download_weights.py
  python download_weights.py --force
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

HF_REPO_ID = "yatoka/LncPNdeep"

SCRIPT_DIR = Path(__file__).resolve().parent

RNA_WEIGHT_FILENAMES = (
    "save.Longformer.pretrain.epoch20.params",
    "save.bigbird.pretrain.epoch20.params",
    "save.bigbird_full.pretrain.epoch20.params",
)
FINAL_CLASSIFIER_NAME = "ProteinTransAllfeature_ResCNN2_07_08.h5"


def get_pretrain_code_dir(base_dir: Path | None = None) -> Path:
    """Return directory containing RNA model definitions (for imports)."""
    root = base_dir or SCRIPT_DIR
    for folder_name in ("simulation_and_pretrain_code", "Nucleotide Embedding"):
        candidate = root / folder_name
        if candidate.is_dir() and (candidate / "model").is_dir():
            return candidate
    return root


# Backward-compatible alias used by inference scripts.
get_nucleotide_embedding_dir = get_pretrain_code_dir


def _weights_pretrain_search_dirs(base_dir: Path) -> List[Path]:
    return [
        base_dir / "simulation_and_pretrain_code" / "weights" / "pretrain",
        base_dir / "Nucleotide Embedding" / "weights" / "pretrain",
        base_dir / "weights" / "pretrain",
    ]


def get_default_weights_pretrain_dir(base_dir: Path | None = None) -> Path:
    """Canonical directory where download_weights.py saves RNA checkpoints."""
    root = base_dir or SCRIPT_DIR
    return root / "simulation_and_pretrain_code" / "weights" / "pretrain"


def get_weights_pretrain_dir(base_dir: Path | None = None) -> Path:
    """Return an existing pretrain weights directory, or the default download target."""
    root = base_dir or SCRIPT_DIR
    for directory in _weights_pretrain_search_dirs(root):
        if directory.is_dir() and any(directory.glob("*.params")):
            return directory
    return get_default_weights_pretrain_dir(root)


def resolve_rna_weight_path(filename: str, base_dir: Path | None = None) -> Path | None:
    root = base_dir or SCRIPT_DIR
    for directory in _weights_pretrain_search_dirs(root):
        path = directory / filename
        if path.is_file():
            return path
    return None


def get_weight_targets(base_dir: Path | None = None) -> List[Tuple[str, Path]]:
    """
    Map Hugging Face repo paths to local paths expected by the prediction scripts.

    Returns list of (hf_filename, local_path).
    """
    root = base_dir or SCRIPT_DIR
    pretrain_dir = get_default_weights_pretrain_dir(root)
    return [
        (
            "weights/rna_pretrain/save.Longformer.pretrain.epoch20.params",
            pretrain_dir / "save.Longformer.pretrain.epoch20.params",
        ),
        (
            "weights/rna_pretrain/save.bigbird.pretrain.epoch20.params",
            pretrain_dir / "save.bigbird.pretrain.epoch20.params",
        ),
        (
            "weights/rna_pretrain/save.bigbird_full.pretrain.epoch20.params",
            pretrain_dir / "save.bigbird_full.pretrain.epoch20.params",
        ),
        (
            "weights/final_classifier/ProteinTransAllfeature_ResCNN2_07_08.h5",
            root / FINAL_CLASSIFIER_NAME,
        ),
    ]


def weights_ready(base_dir: Path | None = None) -> bool:
    """Return True if all required weight files exist locally."""
    root = base_dir or SCRIPT_DIR
    if not (root / FINAL_CLASSIFIER_NAME).is_file():
        return False
    return all(resolve_rna_weight_path(name, root) is not None for name in RNA_WEIGHT_FILENAMES)


def missing_weight_files(base_dir: Path | None = None) -> List[Path]:
    """Return canonical local paths that are missing (for user-facing messages)."""
    return [path for _, path in get_weight_targets(base_dir) if not path.is_file()]


def download_weights(
    base_dir: Path | None = None,
    force: bool = False,
    repo_id: str = HF_REPO_ID,
) -> Dict[str, Path]:
    """
    Download all LncPNdeep weights from Hugging Face into local paths.

    Returns a dict mapping each Hugging Face filename to the local file path.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install it with: pip install huggingface_hub"
        ) from exc

    root = base_dir or SCRIPT_DIR
    downloaded: Dict[str, Path] = {}

    for hf_filename, local_path in get_weight_targets(root):
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.is_file() and not force:
            print(f"Already present, skipping: {local_path}")
            downloaded[hf_filename] = local_path
            continue

        print(f"Downloading {hf_filename} ...")
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            repo_type="model",
        )

        if force and local_path.is_file():
            local_path.unlink()

        if Path(cached_path).resolve() != local_path.resolve():
            # Copy from HF cache to project-relative paths used by the pipeline.
            import shutil

            shutil.copy2(cached_path, local_path)

        print(f"Saved to: {local_path}")
        downloaded[hf_filename] = local_path

    return downloaded


def ensure_weights(
    base_dir: Path | None = None,
    force: bool = False,
    repo_id: str = HF_REPO_ID,
) -> None:
    """Download weights only if any required file is missing (unless force=True)."""
    if force or not weights_ready(base_dir):
        download_weights(base_dir=base_dir, force=force, repo_id=repo_id)
    else:
        print("All required weight files are already present.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download LncPNdeep weights from Hugging Face Hub"
    )
    parser.add_argument(
        "--base_dir",
        default=str(SCRIPT_DIR),
        help="Project root directory (default: directory containing this script)",
    )
    parser.add_argument(
        "--repo_id",
        default=HF_REPO_ID,
        help=f"Hugging Face model repo id (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check whether weights exist; do not download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()

    if args.check:
        if weights_ready(base_dir):
            print("All required weight files are present.")
            for _, path in get_weight_targets(base_dir):
                print(f"  OK  {path}")
        else:
            print("Missing weight files:")
            for path in missing_weight_files(base_dir):
                print(f"  MISSING  {path}")
            raise SystemExit(1)
        return

    ensure_weights(base_dir=base_dir, force=args.force, repo_id=args.repo_id)
    print("Weight download complete.")
    print(f"Hugging Face repo: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
