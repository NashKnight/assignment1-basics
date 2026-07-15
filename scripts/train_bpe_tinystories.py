#!/usr/bin/env python3
"""Train a BPE vocabulary on TinyStories and save it as portable JSON.

Example:
    uv run python scripts/train_bpe_tinystories.py --vocab-size 10000

The output JSON stores every byte sequence as Base64, so it can be loaded
without pickle when implementing the tokenizer's encode/decode methods.
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cs336_basics.train_bpe import train_bpe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"


def encode_bytes(value: bytes) -> str:
    """Return an ASCII-safe representation of arbitrary token bytes."""
    return base64.b64encode(value).decode("ascii")


def save_model(
    output_path: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    input_path: Path,
    special_tokens: list[str],
) -> None:
    """Write a self-contained BPE artifact that future tokenizer code can load."""
    artifact: dict[str, Any] = {
        "format_version": 1,
        "model_type": "byte_level_bpe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_corpus": str(input_path.resolve()),
        "training_corpus_bytes": input_path.stat().st_size,
        "special_tokens": special_tokens,
        "vocab": [
            {"id": token_id, "bytes_base64": encode_bytes(token_bytes)}
            for token_id, token_bytes in sorted(vocab.items())
        ],
        "merges": [
            {"left_base64": encode_bytes(left), "right_base64": encode_bytes(right)}
            for left, right in merges
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE model on TinyStories.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Training corpus (default: {DEFAULT_INPUT})",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Final vocabulary size.")
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=None,
        help="Special token to preserve (repeatable; default: <|endoftext|>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Model JSON path (default: artifacts/tinystories_bpe_<vocab_size>.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    special_tokens = args.special_tokens or ["<|endoftext|>"]
    minimum_vocab_size = 256 + len(special_tokens)

    if not input_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {input_path}")
    if args.vocab_size < minimum_vocab_size:
        raise ValueError(f"--vocab-size must be at least {minimum_vocab_size}.")

    output_path = args.output or (
        PROJECT_ROOT / "artifacts" / f"tinystories_bpe_{args.vocab_size}.json"
    )
    output_path = output_path.resolve()
    merge_count = args.vocab_size - minimum_vocab_size

    print(f"Corpus: {input_path} ({input_path.stat().st_size / 1024**3:.2f} GiB)")
    print("Pre-tokenizing corpus with 4 worker processes; merge progress starts afterwards.")
    with tqdm(total=merge_count, desc="BPE merges", unit="merge", dynamic_ncols=True) as progress:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=args.vocab_size,
            special_tokens=special_tokens,
            progress_callback=lambda _completed, _total: progress.update(1),
        )

    save_model(output_path, vocab, merges, input_path, special_tokens)
    print(f"Saved {len(vocab)} vocabulary entries and {len(merges)} merges to:")
    print(output_path)


if __name__ == "__main__":
    main()
