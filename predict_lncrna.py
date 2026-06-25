# -*- coding: utf-8 -*-
"""
Full lncRNA / coding RNA prediction pipeline:
  1. Input FASTA
  2. Extract nucleotide embeddings (Longformer256, Bigbird256, Bigbird768)
  3. Extract peptide embeddings (Average / Fake / Max)
  4. Load h5 classifier and predict
  5. Write results txt: sequence name + lncRNA probability + label

Example:
  py predict_lncrna.py --input_fasta input.fasta --output_dir work/ --result_txt results.txt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from extract_nucleotide_embeddings import (
    MODEL_CONFIGS,
    extract_embeddings_from_fasta,
    parse_fasta,
)
from download_weights import (
    get_nucleotide_embedding_dir,
    missing_weight_files,
    weights_ready,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_H5 = SCRIPT_DIR / "ProteinTransAllfeature_ResCNN2_07_08.h5"
DEFAULT_WEIGHTS_DIR = get_nucleotide_embedding_dir(SCRIPT_DIR) / "weights" / "pretrain"

_PROT_BERT_TOKENIZER = None
_PROT_BERT_MODEL = None


def _get_protein_bert(device: torch.device):
    global _PROT_BERT_TOKENIZER, _PROT_BERT_MODEL
    if _PROT_BERT_TOKENIZER is None or _PROT_BERT_MODEL is None:
        _PROT_BERT_TOKENIZER = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert", do_lower_case=False
        )
        _PROT_BERT_MODEL = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
        _PROT_BERT_MODEL.eval()
    return _PROT_BERT_TOKENIZER, _PROT_BERT_MODEL


def get_bert_embedding(sequence: str, len_seq_limit: int, device: torch.device) -> np.ndarray:
    tokenizer, model = _get_protein_bert(device)
    sequence_w_spaces = " ".join(list(sequence))
    encoded_input = tokenizer(
        sequence_w_spaces,
        truncation=True,
        max_length=len_seq_limit,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model(**encoded_input)
        output_hidden = output["last_hidden_state"][:, 0][0].detach().cpu().numpy()

    if len(output_hidden) != 1024:
        raise ValueError(f"ProtBERT embedding dimension must be 1024, got {len(output_hidden)}")
    return output_hidden


def extract_protein_embeddings_from_fasta(
    fasta_path: str,
    output_dir: str,
    prefix: str = "protein",
    device: str | None = None,
) -> Tuple[str, str, str]:
    """
    Extract three protein embedding variants from FASTA and save as npy files.

    Returns: (average_path, fake_path, max_path)
    """
    resolved_device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(output_dir, exist_ok=True)

    embed_average: List[np.ndarray] = []
    embed_fake: List[np.ndarray] = []
    embed_max: List[np.ndarray] = []

    for item in tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Extracting protein embeddings"):
        reading_frames = [
            Seq(item.seq).translate(table="Standard", stop_symbol="*", to_stop=False, cds=False)
            for _ in range(3)
        ]
        peptides: List[str] = []
        lengths: List[int] = []
        frame_embeddings: List[np.ndarray] = []

        for frame in reading_frames:
            for peptide in frame.split("*"):
                if len(peptide) > 100:
                    peptides.append(peptide)
                    lengths.append(len(peptide))
                    frame_embeddings.append(
                        get_bert_embedding(
                            sequence=str(peptide),
                            len_seq_limit=1200,
                            device=resolved_device,
                        )
                    )

        if len(peptides) == 0:
            fallback = get_bert_embedding(
                sequence=str(Seq(item.seq).translate()),
                len_seq_limit=1200,
                device=resolved_device,
            )
            embed_average.append(fallback)
            embed_fake.append(fallback)
            embed_max.append(fallback)
        else:
            embed_max.append(frame_embeddings[int(np.argmax(lengths))])
            embed_fake.append(
                get_bert_embedding(
                    sequence=str(Seq(item.seq).translate()),
                    len_seq_limit=1200,
                    device=resolved_device,
                )
            )
            embed_average.append(np.sum(frame_embeddings, axis=0))

    average_path = os.path.join(output_dir, f"{prefix}_Average_Protein.npy")
    fake_path = os.path.join(output_dir, f"{prefix}_Fake_Protein.npy")
    max_path = os.path.join(output_dir, f"{prefix}_Max_Protein.npy")

    np.save(average_path, np.array(embed_average))
    np.save(fake_path, np.array(embed_fake))
    np.save(max_path, np.array(embed_max))

    return average_path, fake_path, max_path


def load_embedding_txt(path: str) -> np.ndarray:
    data = pd.read_table(path, sep=" ", header=None)
    array = np.array(data, dtype=np.float32)
    return array.reshape(array.shape[0], 1, array.shape[1])


def load_protein_npy(path: str) -> np.ndarray:
    array = np.load(path)
    return array.reshape(array.shape[0], 1, array.shape[1])


def predict_lncrna_probability(
    model_h5: str,
    average_protein: np.ndarray,
    fake_protein: np.ndarray,
    max_protein: np.ndarray,
    bigbird256: np.ndarray,
    bigbird768: np.ndarray,
    longformer256: np.ndarray,
) -> np.ndarray:
    """Load h5 model and return softmax probabilities with shape (N, 2): col 0 = lncRNA, col 1 = coding RNA."""
    model = tf.keras.models.load_model(model_h5, compile=False)
    predictions = model.predict(
        [
            average_protein,
            fake_protein,
            max_protein,
            bigbird256,
            bigbird768,
            longformer256,
        ],
        verbose=0,
    )
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        return predictions
    raise ValueError(
        f"Expected model output shape (N, 2) softmax probabilities, got {predictions.shape}"
    )


def run_pipeline(
    input_fasta: str,
    output_dir: str,
    result_txt: str,
    model_h5: str,
    weights_dir: str,
    vocab_path: str | None = None,
    device: str | None = None,
    attention_mode: str = "sliding_chunks",
    skip_nucleotide: bool = False,
    skip_protein: bool = False,
) -> None:
    if not weights_ready(SCRIPT_DIR):
        missing = "\n".join(f"  - {path}" for path in missing_weight_files(SCRIPT_DIR))
        raise FileNotFoundError(
            "Required model weights are missing. Run this first:\n"
            "  python download_weights.py\n\n"
            f"Missing files:\n{missing}"
        )

    os.makedirs(output_dir, exist_ok=True)
    nucleotide_dir = os.path.join(output_dir, "nucleotide")
    protein_dir = os.path.join(output_dir, "protein")

    records = parse_fasta(input_fasta)
    sequence_names = [name for name, _ in records]

    if not skip_nucleotide:
        nucleotide_paths = extract_embeddings_from_fasta(
            fasta_path=input_fasta,
            output_dir=nucleotide_dir,
            models=list(MODEL_CONFIGS.keys()),
            weights_dir=weights_dir,
            vocab_path=vocab_path,
            device=device,
            attention_mode=attention_mode,
        )
    else:
        nucleotide_paths = {
            name: os.path.join(nucleotide_dir, cfg["output_file"])
            for name, cfg in MODEL_CONFIGS.items()
        }
        for path in nucleotide_paths.values():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing nucleotide embedding file: {path}")

    if not skip_protein:
        average_path, fake_path, max_path = extract_protein_embeddings_from_fasta(
            fasta_path=input_fasta,
            output_dir=protein_dir,
            prefix="query",
            device=device,
        )
    else:
        average_path = os.path.join(protein_dir, "query_Average_Protein.npy")
        fake_path = os.path.join(protein_dir, "query_Fake_Protein.npy")
        max_path = os.path.join(protein_dir, "query_Max_Protein.npy")
        for path in (average_path, fake_path, max_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing protein embedding file: {path}")

    probabilities = predict_lncrna_probability(
        model_h5=model_h5,
        average_protein=load_protein_npy(average_path),
        fake_protein=load_protein_npy(fake_path),
        max_protein=load_protein_npy(max_path),
        bigbird256=load_embedding_txt(nucleotide_paths["Bigbird256"]),
        bigbird768=load_embedding_txt(nucleotide_paths["Bigbird768"]),
        longformer256=load_embedding_txt(nucleotide_paths["Longformer256"]),
    )

    if len(probabilities) != len(sequence_names):
        raise ValueError(
            f"Prediction count ({len(probabilities)}) does not match sequence count ({len(sequence_names)})"
        )

    os.makedirs(os.path.dirname(result_txt) or ".", exist_ok=True)
    with open(result_txt, "w", encoding="utf-8") as handle:
        handle.write(
            "sequence_name\tlncRNA_probability\tcodingRNA_probability\tis_lncRNA\n"
        )
        for name, prob_pair in zip(sequence_names, probabilities):
            lnc_prob = float(prob_pair[0])
            coding_prob = float(prob_pair[1])
            is_lnc = "lncRNA" if lnc_prob >= coding_prob else "codingRNA"
            handle.write(
                f"{name}\t{lnc_prob:.6f}\t{coding_prob:.6f}\t{is_lnc}\n"
            )

    print(f"Prediction finished for {len(sequence_names)} sequence(s)")
    print(f"Results written to: {result_txt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full lncRNA / coding RNA prediction pipeline")
    parser.add_argument("--input_fasta", required=True, help="Path to input FASTA file")
    parser.add_argument("--output_dir", required=True, help="Directory for intermediate embeddings and outputs")
    parser.add_argument(
        "--result_txt",
        default=None,
        help="Path to final results txt (default: output_dir/prediction_results.txt)",
    )
    parser.add_argument(
        "--model_h5",
        default=str(DEFAULT_MODEL_H5),
        help="Path to final classification model h5 file",
    )
    parser.add_argument(
        "--weights_dir",
        default=str(DEFAULT_WEIGHTS_DIR),
        help="Directory containing nucleotide pretrain weight files",
    )
    parser.add_argument(
        "--vocab_path",
        default=None,
        help="Optional path to pre_valid_kmer1.txt for vocabulary",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Inference device (default: auto)")
    parser.add_argument(
        "--attention_mode",
        default="sliding_chunks",
        choices=["sliding_chunks", "n2", "tvm"],
        help="Longformer attention mode",
    )
    parser.add_argument(
        "--skip_nucleotide",
        action="store_true",
        help="Skip nucleotide embedding extraction (requires existing files in output_dir/nucleotide)",
    )
    parser.add_argument(
        "--skip_protein",
        action="store_true",
        help="Skip protein embedding extraction (requires existing npy files in output_dir/protein)",
    )
    parser.add_argument(
        "--download_weights",
        action="store_true",
        help="Download missing weights from Hugging Face before running",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.download_weights:
        from download_weights import ensure_weights

        ensure_weights(base_dir=SCRIPT_DIR)
    result_txt = args.result_txt or os.path.join(args.output_dir, "prediction_results.txt")
    run_pipeline(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        result_txt=result_txt,
        model_h5=args.model_h5,
        weights_dir=args.weights_dir,
        vocab_path=args.vocab_path,
        device=args.device,
        attention_mode=args.attention_mode,
        skip_nucleotide=args.skip_nucleotide,
        skip_protein=args.skip_protein,
    )


if __name__ == "__main__":
    main()
