#!/usr/bin/env python3
"""Run tokenizer experiments and save measurements only."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from cs336_basics.Tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_OUTPUT = DEFAULT_ARTIFACTS_DIR / "tokenizer_experiments" / "results.json"
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
PILE_BYTES = 825 * 1000**3


def load_tokenizer(
    artifacts_dir: Path,
    dataset: str,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    dataset_dir = artifacts_dir / dataset
    return Tokenizer.from_files(
        vocab_filepath=dataset_dir / "vocab.jsonl",
        merges_filepath=dataset_dir / "merges.jsonl",
        special_tokens=special_tokens,
    )


def iter_documents(
    path: Path,
    delimiter: str = "<|endoftext|>",
    chunk_size: int = 8 * 1024 * 1024,
) -> Iterable[str]:
    buffer = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while chunk := f.read(chunk_size):
            buffer += chunk
            parts = buffer.split(delimiter)
            yield from (part for part in parts[:-1] if part)
            buffer = parts[-1]
    if buffer:
        yield buffer


def sample_documents(
    path: Path,
    num_documents: int,
    strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = []
    seen = 0

    for document_index, text in enumerate(iter_documents(path)):
        if not text:
            continue
        seen += 1
        sample = {"document_index": document_index, "text": text}

        if strategy == "first":
            samples.append(sample)
            if len(samples) == num_documents:
                break
            continue

        if len(samples) < num_documents:
            samples.append(sample)
        else:
            replacement_index = rng.randrange(seen)
            if replacement_index < num_documents:
                samples[replacement_index] = sample

    if len(samples) < num_documents:
        raise ValueError(f"Only found {len(samples)} documents in {path}, expected {num_documents}.")
    return samples


def compression_measurement(
    dataset: str,
    tokenizer_name: str,
    tokenizer: Tokenizer,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    documents = []
    total_bytes = 0
    total_tokens = 0

    for sample in samples:
        text = str(sample["text"])
        num_bytes = len(text.encode("utf-8"))
        num_tokens = len(tokenizer.encode(text))
        total_bytes += num_bytes
        total_tokens += num_tokens
        documents.append(
            {
                "document_index": sample["document_index"],
                "bytes": num_bytes,
                "tokens": num_tokens,
                "bytes_per_token": num_bytes / num_tokens if num_tokens else None,
            }
        )

    return {
        "dataset": dataset,
        "tokenizer": tokenizer_name,
        "num_documents": len(samples),
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else None,
        "documents": documents,
    }


def read_text_prefix(path: Path, byte_limit: int) -> str:
    with path.open("rb") as f:
        raw = f.read(byte_limit)
    return raw.decode("utf-8", errors="replace")


def throughput_measurement(
    dataset: str,
    tokenizer_name: str,
    tokenizer: Tokenizer,
    corpus_path: Path,
    byte_limit: int,
) -> dict[str, Any]:
    text = read_text_prefix(corpus_path, byte_limit)
    input_bytes = len(text.encode("utf-8"))

    start = time.perf_counter()
    token_ids = tokenizer.encode(text)
    elapsed_seconds = time.perf_counter() - start
    bytes_per_second = input_bytes / elapsed_seconds if elapsed_seconds else None

    return {
        "dataset": dataset,
        "tokenizer": tokenizer_name,
        "corpus_path": str(corpus_path),
        "input_bytes": input_bytes,
        "tokens": len(token_ids),
        "elapsed_seconds": elapsed_seconds,
        "bytes_per_second": bytes_per_second,
        "pile_825gb_seconds": PILE_BYTES / bytes_per_second if bytes_per_second else None,
        "pile_825gb_hours": PILE_BYTES / bytes_per_second / 3600 if bytes_per_second else None,
    }


def run_experiments(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts_dir.resolve()
    special_tokens = args.special_tokens or DEFAULT_SPECIAL_TOKENS
    tinystories_path = args.tinystories_corpus.resolve()
    owt_path = args.owt_corpus.resolve()

    tokenizers = {
        "tinystories": load_tokenizer(artifacts_dir, "tinystories", special_tokens),
        "owt": load_tokenizer(artifacts_dir, "owt", special_tokens),
    }
    samples = {
        "tinystories": sample_documents(tinystories_path, args.num_documents, args.sample_strategy, args.seed),
        "owt": sample_documents(owt_path, args.num_documents, args.sample_strategy, args.seed),
    }

    compression = [
        compression_measurement("tinystories", "tinystories", tokenizers["tinystories"], samples["tinystories"]),
        compression_measurement("owt", "owt", tokenizers["owt"], samples["owt"]),
        compression_measurement("owt", "tinystories", tokenizers["tinystories"], samples["owt"]),
    ]

    throughput = []
    if args.throughput_bytes > 0:
        throughput.extend(
            [
                throughput_measurement(
                    "tinystories",
                    "tinystories",
                    tokenizers["tinystories"],
                    tinystories_path,
                    args.throughput_bytes,
                ),
                throughput_measurement("owt", "owt", tokenizers["owt"], owt_path, args.throughput_bytes),
            ]
        )

    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_documents": args.num_documents,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "special_tokens": special_tokens,
        "compression": compression,
        "throughput": throughput,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote tokenizer experiment results to {args.output}")


def count_tokens(tokenizer: Tokenizer, input_path: Path) -> int:
    token_count = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for _token_id in tqdm(tokenizer.encode_iterable(f), desc="count tokens", unit="token"):
            token_count += 1
    return token_count


def write_token_ids(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    token_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_array = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(token_count,))

    offset = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for token_id in tqdm(tokenizer.encode_iterable(f), total=token_count, desc="write token ids", unit="token"):
            if token_id > np.iinfo(np.uint16).max:
                raise ValueError(f"Token id {token_id} does not fit in uint16.")
            token_array[offset] = token_id
            offset += 1
    token_array.flush()


def encode_dataset(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts_dir.resolve()
    special_tokens = args.special_tokens or DEFAULT_SPECIAL_TOKENS
    tokenizer = load_tokenizer(artifacts_dir, args.tokenizer, special_tokens)
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    metadata_path = args.metadata.resolve() if args.metadata else output_path.with_suffix(f"{output_path.suffix}.json")

    start = time.perf_counter()
    token_count = count_tokens(tokenizer, input_path)
    write_token_ids(tokenizer, input_path, output_path, token_count)
    elapsed_seconds = time.perf_counter() - start

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "output_path": str(output_path),
        "dtype": "uint16",
        "tokenizer": args.tokenizer,
        "token_count": token_count,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": token_count / elapsed_seconds if elapsed_seconds else None,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote token ids to {output_path}")
    print(f"Wrote encoding metadata to {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tokenizer experiments without generating report answers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Measure compression ratios and tokenizer throughput.")
    run_parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    run_parser.add_argument(
        "--tinystories-corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
    )
    run_parser.add_argument("--owt-corpus", type=Path, default=PROJECT_ROOT / "data" / "owt_train.txt")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--num-documents", type=int, default=10)
    run_parser.add_argument("--sample-strategy", choices=["first", "reservoir"], default="first")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--throughput-bytes", type=int, default=1_000_000)
    run_parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)
    run_parser.set_defaults(func=run_experiments)

    encode_parser = subparsers.add_parser("encode", help="Encode one dataset into a uint16 .npy file.")
    encode_parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    encode_parser.add_argument("--tokenizer", choices=["tinystories", "owt"], required=True)
    encode_parser.add_argument("--input", type=Path, required=True)
    encode_parser.add_argument("--output", type=Path, required=True)
    encode_parser.add_argument("--metadata", type=Path, default=None)
    encode_parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)
    encode_parser.set_defaults(func=encode_dataset)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
