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


def get_nucleotide_embedding_dir(base_dir: Path | None = None) -> Path:
    """Return nucleotide embedding code directory (supports flat or nested layouts)."""
    root = base_dir or SCRIPT_DIR
    nested = root / "Nucleotide Embedding"
    if nested.is_dir():
        return nested
    return root


def get_weight_targets(base_dir: Path | None = None) -> List[Tuple[str, Path]]:
    """
    Map Hugging Face repo paths to local paths expected by the prediction scripts.

    Returns list of (hf_filename, local_path).
    """
    root = base_dir or SCRIPT_DIR
    pretrain_dir = get_nucleotide_embedding_dir(root) / "weights" / "pretrain"
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
            root / "ProteinTransAllfeature_ResCNN2_07_08.h5",
        ),
    ]


def weights_ready(base_dir: Path | None = None) -> bool:
    """Return True if all required weight files exist locally."""
    return all(path.is_file() for _, path in get_weight_targets(base_dir))


def missing_weight_files(base_dir: Path | None = None) -> List[Path]:
    """Return local paths that are missing."""
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
