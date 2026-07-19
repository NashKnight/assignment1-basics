#!/usr/bin/env python3
"""Train BPE tokenizers and store reproducible results as per-dataset JSONL files."""

from __future__ import annotations

import argparse
import base64
import json
import re
import resource
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LEGACY_COMBINED_RESULTS = DEFAULT_ARTIFACTS_DIR / "bpe_training_results.jsonl"

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


def decode_bytes(value: str) -> bytes:
    return base64.b64decode(value)


def parse_elapsed_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None
    return None


def parse_time_value(log_text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.+)$", log_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def find_last_matching_line(text: str, pattern: str) -> str | None:
    regex = re.compile(pattern)
    lines = text.replace("\r", "\n").splitlines()
    matches = [line for line in lines if regex.search(line)]
    return matches[-1] if matches else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    tmp_path.replace(path)


def dataset_dir(artifacts_dir: Path, dataset: str) -> Path:
    return artifacts_dir / dataset


def dataset_paths(artifacts_dir: Path, dataset: str) -> dict[str, Path]:
    root = dataset_dir(artifacts_dir, dataset)
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
        if record_type in {"run", "answer"}:
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


def migrate_legacy_combined_results(artifacts_dir: Path, legacy_path: Path = LEGACY_COMBINED_RESULTS) -> None:
    """Split the old monolithic JSONL into per-dataset run/vocab/merges files once."""
    if not legacy_path.exists():
        return

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in load_jsonl(legacy_path):
        dataset = str(record.get("dataset", "")).strip()
        if not dataset:
            continue
        by_dataset.setdefault(dataset, []).append(record)

    for dataset, records in by_dataset.items():
        paths = dataset_paths(artifacts_dir, dataset)
        if paths["run"].exists() or paths["vocab"].exists() or paths["merges"].exists():
            continue
        save_dataset_artifacts(artifacts_dir, dataset, records)
        print(f"Migrated legacy records for {dataset} -> {dataset_dir(artifacts_dir, dataset)}")

    backup_path = legacy_path.with_name(f"{legacy_path.name}.legacy_backup")
    if not backup_path.exists():
        legacy_path.replace(backup_path)
        print(f"Renamed {legacy_path} -> {backup_path}")


def decode_vocab_entry(entry: dict[str, Any]) -> bytes:
    return decode_bytes(str(entry["bytes_base64"]))


def metrics_from_log(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {}

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    elapsed = parse_time_value(log_text, "Elapsed (wall clock) time (h:mm:ss or m:ss)")
    user_time = parse_time_value(log_text, "User time (seconds)")
    system_time = parse_time_value(log_text, "System time (seconds)")
    max_rss_kb_raw = parse_time_value(log_text, "Maximum resident set size (kbytes)")

    metrics: dict[str, Any] = {
        "log_path": str(log_path.resolve()),
        "elapsed_wall_clock_time": elapsed,
        "elapsed_seconds": parse_elapsed_seconds(elapsed),
        "user_time_seconds": float(user_time) if user_time is not None else None,
        "system_time_seconds": float(system_time) if system_time is not None else None,
        "final_merge_progress_line": find_last_matching_line(log_text, r"BPE merges:.*100%"),
    }
    if max_rss_kb_raw is not None:
        max_rss_kb = int(max_rss_kb_raw)
        metrics.update(
            {
                "maximum_resident_set_size_kbytes": max_rss_kb,
                "maximum_resident_set_size_mib": round(max_rss_kb / 1024, 1),
                "maximum_resident_set_size_gib": round(max_rss_kb / (1024 * 1024), 2),
            }
        )
    return metrics


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
    records.extend(answer_records(run_record, artifact_paths))
    return records


def answer_records(
    run: dict[str, Any],
    artifact_paths: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    dataset = str(run["dataset"])
    if artifact_paths is not None:
        location = (
            f"`{artifact_relpath(artifact_paths['run'])}`, "
            f"`{artifact_relpath(artifact_paths['vocab'])}`, and "
            f"`{artifact_relpath(artifact_paths['merges'])}`"
        )
    else:
        location = f"`artifacts/{dataset}/{{run,vocab,merges}}.jsonl`"

    if dataset == "tinystories":
        answer_a = (
            "I trained a byte-level BPE tokenizer on TinyStories with maximum vocabulary size "
            f"{run['vocab_size']:,} and added `<|endoftext|>` as a special token. The run summary, "
            f"vocabulary, and merges were serialized to {location}; training took "
            f"{run.get('elapsed_wall_clock_time', 'N/A')} wall-clock time and used "
            f"{run.get('maximum_resident_set_size_kbytes', 'N/A')} KB of peak resident memory "
            f"(about {run.get('maximum_resident_set_size_gib', 'N/A')} GiB), and the longest token is "
            f"`{run['longest_token_bytes_repr']}` ({run['longest_token_num_bytes']} bytes), which decodes to "
            f"`{run['longest_token_utf8']!r}`. This makes sense because common English word pieces, "
            "especially words with a leading space, are frequent enough in TinyStories to become single "
            "BPE tokens."
        )
        answer_b = (
            "The slowest part of my tokenizer training is the BPE merge loop: each merge iteration "
            "updates pair statistics for pre-tokens affected by the chosen merge."
        )
        return [
            answer_record(run, "a", answer_a),
            answer_record(run, "b", answer_b),
        ]

    if dataset == "owt":
        answer_a = (
            "I trained a byte-level BPE tokenizer on OpenWebText with maximum vocabulary size "
            f"{run['vocab_size']:,} and serialized the run summary, vocabulary, and merges to "
            f"{location}. The longest token is "
            f"`{run['longest_token_bytes_repr']}` ({run['longest_token_num_bytes']} bytes), which decodes to "
            f"`{run['longest_token_utf8']!r}`; this is plausible for OpenWebText because web text contains "
            "long repeated fragments, formatting artifacts, URLs, and domain-specific strings."
        )
        return [answer_record(run, "a", answer_a)]

    return []


def answer_record(run: dict[str, Any], part: str, text: str) -> dict[str, Any]:
    return {
        "record_type": "answer",
        "dataset": run["dataset"],
        "problem": run["problem"],
        "part": part,
        "text": text,
    }


def import_artifact(args: argparse.Namespace) -> None:
    artifacts_dir = args.artifacts_dir.resolve()
    migrate_legacy_combined_results(artifacts_dir)

    artifact_path = args.artifact.resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    vocab = {
        int(entry["id"]): decode_vocab_entry(entry)
        for entry in artifact["vocab"]
    }
    merges = [
        (decode_bytes(entry["left_base64"]), decode_bytes(entry["right_base64"]))
        for entry in artifact["merges"]
    ]
    input_path = Path(artifact["training_corpus"])
    metrics = metrics_from_log(args.log.resolve() if args.log else None)
    metrics["source_artifact"] = str(artifact_path)
    paths = dataset_paths(artifacts_dir, args.dataset)
    records = build_records(
        dataset=args.dataset,
        problem=args.problem or DATASETS[args.dataset]["problem"],
        input_path=input_path,
        vocab=vocab,
        merges=merges,
        special_tokens=artifact["special_tokens"],
        metrics=metrics,
        artifact_paths=paths,
    )
    saved = save_dataset_artifacts(artifacts_dir, args.dataset, records)
    print(
        f"Wrote {len(records)} records for {args.dataset} to "
        f"{saved['run']}, {saved['vocab']}, {saved['merges']}"
    )


def train_dataset(args: argparse.Namespace) -> None:
    from cs336_basics.train_bpe import train_bpe

    artifacts_dir = args.artifacts_dir.resolve()
    migrate_legacy_combined_results(artifacts_dir)

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
    print(f"Saved run/answer summary to {saved['run']}")
    print(f"elapsed_wall_clock_time: {metrics['elapsed_wall_clock_time']}")
    print(f"maximum_resident_set_size_kbytes: {peak_rss_kb}")


def format_seconds(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:05.2f}"
    return f"{minutes}:{sec:05.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE tokenizers and maintain JSONL artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a dataset BPE and write JSONL records.")
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
    # Backward-compatible alias; if someone still passes --results, treat it as artifacts root.
    train_parser.add_argument("--results", type=Path, default=None, help=argparse.SUPPRESS)
    train_parser.set_defaults(func=train_dataset)

    import_parser = subparsers.add_parser("import-artifact", help="Import an old JSON artifact into JSONL.")
    import_parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    import_parser.add_argument("--artifact", type=Path, required=True)
    import_parser.add_argument("--log", type=Path, default=None)
    import_parser.add_argument("--problem", default=None)
    import_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Root directory for per-dataset artifacts (default: artifacts/).",
    )
    import_parser.add_argument("--results", type=Path, default=None, help=argparse.SUPPRESS)
    import_parser.set_defaults(func=import_artifact)

    migrate_parser = subparsers.add_parser(
        "migrate-legacy",
        help="Split artifacts/bpe_training_results.jsonl into per-dataset run/vocab/merges files.",
    )
    migrate_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Root directory for per-dataset artifacts (default: artifacts/).",
    )
    migrate_parser.set_defaults(func=lambda args: migrate_legacy_combined_results(args.artifacts_dir.resolve()))

    args = parser.parse_args()
    if getattr(args, "results", None) is not None and getattr(args, "artifacts_dir", None) is not None:
        # Prefer explicit --results only when user still uses the old flag.
        if args.results is not None:
            args.artifacts_dir = args.results
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
