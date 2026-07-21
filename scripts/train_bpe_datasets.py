#!/usr/bin/env python3
"""Train BPE tokenizers and store reproducible results as per-dataset JSONL files."""

from __future__ import annotations

import argparse
import base64
import json
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASETS: dict[str, dict[str, Any]] = {
    "tinystories": {
        "problem": "train_bpe_tinystories",
        "input": PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        "vocab_size": 10_000,
    },
    "owt": {
        "problem": "train_bpe_expts_owt",
        "input": PROJECT_ROOT / "data" / "owt_train.txt",
        "vocab_size": 32_000,
    },
}


def encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_path.replace(path)


def dataset_paths(artifacts_dir: Path, dataset: str) -> dict[str, Path]:
    root = artifacts_dir / dataset
    return {
        "run": root / "run.jsonl",
        "vocab": root / "vocab.jsonl",
        "merges": root / "merges.jsonl",
    }


def artifact_relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def split_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_records: list[dict[str, Any]] = []
    vocab_records: list[dict[str, Any]] = []
    merge_records: list[dict[str, Any]] = []
    for record in records:
        record_type = record.get("record_type")
        if record_type == "run":
            run_records.append(record)
        elif record_type == "vocab":
            vocab_records.append(record)
        elif record_type == "merge":
            merge_records.append(record)
        else:
            raise ValueError(f"Unknown record_type: {record_type!r}")
    return run_records, vocab_records, merge_records


def save_dataset_artifacts(
    artifacts_dir: Path,
    dataset: str,
    records: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = dataset_paths(artifacts_dir, dataset)
    run_records, vocab_records, merge_records = split_records(records)
    write_jsonl(paths["run"], run_records)
    write_jsonl(paths["vocab"], vocab_records)
    write_jsonl(paths["merges"], merge_records)
    return paths


def current_peak_rss_kb() -> int:
    self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(self_rss, child_rss)


def build_records(
    dataset: str,
    problem: str,
    input_path: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
    metrics: dict[str, Any],
    artifact_paths: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    longest_id, longest_bytes = max(vocab.items(), key=lambda item: len(item[1]))
    run_record: dict[str, Any] = {
        "record_type": "run",
        "dataset": dataset,
        "problem": problem,
        "model_type": "byte_level_bpe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_corpus": str(input_path.resolve()),
        "training_corpus_bytes": input_path.stat().st_size if input_path.exists() else None,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "longest_token_id": longest_id,
        "longest_token_num_bytes": len(longest_bytes),
        "longest_token_bytes_base64": encode_bytes(longest_bytes),
        "longest_token_bytes_repr": repr(longest_bytes),
        "longest_token_utf8": longest_bytes.decode("utf-8", errors="replace"),
    }
    if artifact_paths is not None:
        run_record.update(
            {
                "run_jsonl": artifact_relpath(artifact_paths["run"]),
                "vocab_jsonl": artifact_relpath(artifact_paths["vocab"]),
                "merges_jsonl": artifact_relpath(artifact_paths["merges"]),
            }
        )
    run_record.update({key: value for key, value in metrics.items() if value is not None})

    records = [run_record]
    records.extend(
        {
            "record_type": "vocab",
            "dataset": dataset,
            "token_id": token_id,
            "bytes_base64": encode_bytes(token_bytes),
        }
        for token_id, token_bytes in sorted(vocab.items())
    )
    records.extend(
        {
            "record_type": "merge",
            "dataset": dataset,
            "rank": rank,
            "left_base64": encode_bytes(left),
            "right_base64": encode_bytes(right),
        }
        for rank, (left, right) in enumerate(merges)
    )
    return records


def train_dataset(args: argparse.Namespace) -> None:
    from cs336_basics.train_bpe import train_bpe

    artifacts_dir = args.artifacts_dir.resolve()

    preset = DATASETS[args.dataset]
    input_path = (args.input or preset["input"]).resolve()
    vocab_size = args.vocab_size or preset["vocab_size"]
    special_tokens = args.special_tokens or ["<|endoftext|>"]

    if not input_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {input_path}")

    merge_count = vocab_size - 256 - len(special_tokens)
    print(f"Corpus: {input_path} ({input_path.stat().st_size / 1024**3:.2f} GiB)")
    print(f"Dataset: {args.dataset}; vocab_size={vocab_size}; special_tokens={special_tokens}")
    start = time.perf_counter()
    with tqdm(total=merge_count, desc=f"{args.dataset} BPE merges", unit="merge", dynamic_ncols=True) as progress:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            progress_callback=lambda _done, _total: progress.update(1),
        )
    elapsed_seconds = time.perf_counter() - start
    peak_rss_kb = current_peak_rss_kb()
    metrics = {
        "elapsed_seconds": round(elapsed_seconds, 2),
        "elapsed_wall_clock_time": format_seconds(elapsed_seconds),
        "maximum_resident_set_size_kbytes": peak_rss_kb,
        "maximum_resident_set_size_mib": round(peak_rss_kb / 1024, 1),
        "maximum_resident_set_size_gib": round(peak_rss_kb / (1024 * 1024), 2),
    }
    paths = dataset_paths(artifacts_dir, args.dataset)
    records = build_records(
        dataset=args.dataset,
        problem=preset["problem"],
        input_path=input_path,
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        metrics=metrics,
        artifact_paths=paths,
    )
    saved = save_dataset_artifacts(artifacts_dir, args.dataset, records)
    print(f"Saved {len(vocab)} vocabulary entries to {saved['vocab']}")
    print(f"Saved {len(merges)} merges to {saved['merges']}")
    print(f"Saved run summary to {saved['run']}")
    print(f"elapsed_wall_clock_time: {metrics['elapsed_wall_clock_time']}")
    print(f"maximum_resident_set_size_kbytes: {peak_rss_kb}")


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:05.2f}"
    return f"{minutes}:{sec:05.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE tokenizers and write per-dataset JSONL artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a dataset BPE and write run/vocab/merges JSONL.")
    train_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    train_parser.add_argument("--input", type=Path, default=None)
    train_parser.add_argument("--vocab-size", type=int, default=None)
    train_parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)
    train_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Root directory for per-dataset artifacts (default: artifacts/).",
    )
    train_parser.set_defaults(func=train_dataset)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
