# -*- coding: utf-8 -*-
"""
Extract nucleotide embeddings from FASTA (Longformer256 / Bigbird256 / Bigbird768).

Example:
  py extract_nucleotide_embeddings.py --input_fasta input.fasta --output_dir embeddings/
  py extract_nucleotide_embeddings.py --input_fasta input.fasta --output_dir embeddings/ --model all
  py extract_nucleotide_embeddings.py --input_fasta input.fasta --output_dir embeddings/ --model Bigbird256
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import BigBirdConfig, LongformerConfig

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from download_weights import get_nucleotide_embedding_dir, get_weights_pretrain_dir  # noqa: E402

NUCLEOTIDE_EMBEDDING_DIR = get_nucleotide_embedding_dir(SCRIPT_DIR)
sys.path.insert(0, str(NUCLEOTIDE_EMBEDDING_DIR))

from model.mybigbird import MYConfig, Mymodel  # noqa: E402
from utils import sequence_split  # noqa: E402

random.seed(0)
torch.manual_seed(0)

MASK_CHAR = "\u2047"
PAD_CHAR = "\u25A1"
CLS_CHAR = "[CLS]"

MODEL_CONFIGS = {
    "Longformer256": {
        "model_type": "Longformer",
        "weight_name": "save.Longformer.pretrain.epoch20.params",
        "hidden_size": 256,
        "layers": 4,
        "heads": 8,
        "output_file": "Longformer256_embeddings.txt",
    },
    "Bigbird256": {
        "model_type": "Bigbird",
        "weight_name": "save.bigbird.pretrain.epoch20.params",
        "hidden_size": 256,
        "layers": 4,
        "heads": 8,
        "output_file": "Bigbird256_embeddings.txt",
    },
    "Bigbird768": {
        "model_type": "Bigbird",
        "weight_name": "save.bigbird_full.pretrain.epoch20.params",
        "hidden_size": 768,
        "layers": 12,
        "heads": 12,
        "output_file": "Bigbird768_embeddings.txt",
    },
}


def build_stoi(vocab_path: str | None = None) -> Dict[str, int]:
    if vocab_path and os.path.isfile(vocab_path):
        from pre_dataset import Pre_RNADataset

        dataset = Pre_RNADataset(open(vocab_path, encoding="UTF-8").readlines()[:-1])
        return dataset.stoi

    chars = sorted(["A", "C", "G", "T", "N"])
    chars_unique = chars.copy()
    chars_unique.insert(0, MASK_CHAR)
    chars_unique.insert(0, CLS_CHAR)
    chars_unique.insert(0, PAD_CHAR)
    return {ch: i for i, ch in enumerate(chars_unique)}


def normalize_sequence(sequence: str) -> List[str]:
    sequence = sequence.upper().replace("U", "T")
    tokens: List[str] = []
    for base in sequence:
        if base in {"A", "C", "G", "T"}:
            tokens.append(base)
        else:
            tokens.append("N")
    return tokens


def load_nucleotide_model(
    model_name: str,
    weights_dir: str,
    device: torch.device,
    attention_mode: str = "sliding_chunks",
) -> torch.nn.Module:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown Model: {model_name}. Possible list: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[model_name]
    weight_path = os.path.join(weights_dir, cfg["weight_name"])
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Model Params are Missing: {weight_path}")

    if cfg["model_type"] == "Longformer":
        config = LongformerConfig(
            num_hidden_layers=cfg["layers"],
            num_attention_heads=cfg["heads"],
            hidden_size=cfg["hidden_size"],
            max_position_embeddings=4096,
            vocab_size=8,
            eos_token_id=None,
            sep_token_id=None,
        )
        config.attention_mode = attention_mode
        mconf = MYConfig(
            model="Longformer",
            config=config,
            n_embd=config.hidden_size,
            num_class=8,
            pretrain=True,
        )
    else:
        config = BigBirdConfig(
            num_hidden_layers=cfg["layers"],
            num_attention_heads=cfg["heads"],
            hidden_size=cfg["hidden_size"],
            max_position_embeddings=4096,
            vocab_size=8,
            eos_token_id=None,
            sep_token_id=None,
        )
        mconf = MYConfig(
            model="Bigbird",
            config=config,
            n_embd=config.hidden_size,
            num_class=8,
            pretrain=True,
        )

    model = Mymodel(mconf)
    state_dict = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model.to(device)


def extract_sequence_embedding(
    model: torch.nn.Module,
    tokens: List[str],
    stoi: Dict[str, int],
    device: torch.device,
    repeat: int = 10,
) -> np.ndarray:
    segments = sequence_split(tokens, repeat=repeat)
    features: List[np.ndarray] = []

    with torch.no_grad():
        for segment in segments:
            segment_tokens = [CLS_CHAR] + list(segment)
            input_ids = torch.tensor(
                [stoi.get(token, stoi["N"]) for token in segment_tokens],
                dtype=torch.long,
            ).view(1, -1).to(device)
            labels = torch.zeros(1, input_ids.size(1), dtype=torch.long, device=device)
            attention_mask = torch.ones(1, input_ids.size(1), dtype=torch.long, device=device)

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )
            cls_hidden = outputs.last_h[:, 0, :].detach().squeeze().cpu().numpy()
            features.append(cls_hidden)

    return np.mean(np.vstack(features), axis=0)


def parse_fasta(fasta_path: str) -> List[Tuple[str, List[str]]]:
    records: List[Tuple[str, List[str]]] = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        records.append((record.id, normalize_sequence(str(record.seq))))
    if not records:
        raise ValueError(f"FASTA file is empty or missing: {fasta_path}")
    return records


def write_embedding_file(output_path: str, embeddings: List[np.ndarray]) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for embedding in embeddings:
            line = " ".join(str(value) for value in embedding.tolist())
            handle.write(line + "\n")


def extract_embeddings_from_fasta(
    fasta_path: str,
    output_dir: str,
    models: List[str],
    weights_dir: str,
    vocab_path: str | None = None,
    device: str | None = None,
    attention_mode: str = "sliding_chunks",
) -> Dict[str, str]:
    resolved_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(output_dir, exist_ok=True)

    records = parse_fasta(fasta_path)
    stoi = build_stoi(vocab_path)

    names_path = os.path.join(output_dir, "sequence_names.txt")
    with open(names_path, "w", encoding="utf-8") as handle:
        for name, _ in records:
            handle.write(name + "\n")

    output_paths: Dict[str, str] = {}
    for model_name in models:
        model = load_nucleotide_model(
            model_name=model_name,
            weights_dir=weights_dir,
            device=resolved_device,
            attention_mode=attention_mode,
        )
        embeddings: List[np.ndarray] = []
        for _, tokens in tqdm(records, desc=f"Extracting {model_name}"):
            embeddings.append(
                extract_sequence_embedding(
                    model=model,
                    tokens=tokens,
                    stoi=stoi,
                    device=resolved_device,
                )
            )

        output_path = os.path.join(output_dir, MODEL_CONFIGS[model_name]["output_file"])
        write_embedding_file(output_path, embeddings)
        output_paths[model_name] = output_path

        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Longformer256 / Bigbird256 / Bigbird768 for embeddings"
    )
    parser.add_argument("--input_fasta", required=True, help="Input FASTA Location")
    parser.add_argument("--output_dir", required=True, help="Output Directory Location")
    parser.add_argument(
        "--model",
        default="all",
        choices=["all", "Longformer256", "Bigbird256", "Bigbird768"],
        help="The original LncPNdeep use all embeddings for prediction",
    )
    parser.add_argument(
        "--weights_dir",
        default=str(get_weights_pretrain_dir(SCRIPT_DIR)),
        help="Directory Location for save.*.params",
    )
    parser.add_argument(
        "--vocab_path",
        default=None,
        help="pre_valid_kmer1.txt",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
        help="cpu or cuda",
    )
    parser.add_argument(
        "--attention_mode",
        default="sliding_chunks",
        choices=["sliding_chunks", "n2", "tvm"],
        help="Longformer attention pattern",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    output_paths = extract_embeddings_from_fasta(
        fasta_path=args.input_fasta,
        output_dir=args.output_dir,
        models=models,
        weights_dir=args.weights_dir,
        vocab_path=args.vocab_path,
        device=args.device,
        attention_mode=args.attention_mode,
    )
    print("Nucleotide embeddings finished")
    for model_name, path in output_paths.items():
        print(f"  {model_name}: {path}")
    print(f"  sequence names: {os.path.join(args.output_dir, 'sequence_names.txt')}")


if __name__ == "__main__":
    main()
