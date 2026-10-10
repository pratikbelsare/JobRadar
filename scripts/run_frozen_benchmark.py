"""Run the frozen JobRadar benchmark without DynamoDB, queues, or an LLM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from job_intelligence.ai import build_embedding_provider
from job_intelligence.benchmark import (
    BenchmarkMetrics,
    BenchmarkRow,
    benchmark_fingerprint,
    calculate_benchmark_metrics,
    evaluate_frozen_benchmark,
    load_benchmark_labels,
    load_frozen_benchmark,
    write_json,
)
from job_intelligence.config import EmbeddingProvider, Settings

SUBSETS = ("all", "core", "core_adjacent", "eligible", "stretch", "retained")


def main() -> None:
    args = _arguments()
    benchmark_dir = Path(args.benchmark_dir)
    benchmark = load_frozen_benchmark(benchmark_dir)
    if benchmark.manifest.embedding_provider != EmbeddingProvider.HUGGINGFACE_LOCAL.value:
        raise ValueError("offline benchmark requires a frozen huggingface_local provider")
    settings = Settings(
        embedding_provider=EmbeddingProvider.HUGGINGFACE_LOCAL,
        local_embedding_model_id=benchmark.manifest.embedding_model_id,
        local_embedding_device=benchmark.manifest.embedding_device,
    )
    rows = evaluate_frozen_benchmark(
        benchmark,
        build_embedding_provider(settings),
        verify_frozen_stages=not args.allow_stage_drift,
    )
    raw_count = len(rows)
    if args.collapse_duplicates:
        rows = _collapse_duplicate_groups(rows)
    labels_path = Path(args.labels_file) if args.labels_file else benchmark_dir / "labels.csv"
    labels = load_benchmark_labels(labels_path)
    if {row.job_id for row in rows} != {label.job_id for label in labels}:
        raise ValueError("labels.csv must contain exactly one label row per frozen job")
    metrics = [
        calculate_benchmark_metrics(
            rows,
            labels,
            label_source=args.label_source,
            subset=subset,
        )
        for subset in SUBSETS
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark_version": benchmark.manifest.benchmark_version,
        "profile_version": benchmark.manifest.profile_version,
        "profile_sha256": benchmark.manifest.profile_sha256,
        "semantic_fingerprint": benchmark.manifest.semantic_fingerprint,
        "fingerprint": benchmark_fingerprint(rows),
        "raw_job_count": raw_count,
        "evaluated_job_count": len(rows),
        "duplicates_collapsed": args.collapse_duplicates,
        "frozen_stage_verification": not args.allow_stage_drift,
        "label_source": args.label_source,
        "metrics": [item.model_dump(mode="json") for item in metrics],
        "results": [item.model_dump(mode="json") for item in rows],
    }
    write_json(output_dir / "benchmark-results.json", payload)
    _write_csv(output_dir / "benchmark-results.csv", rows)
    (output_dir / "benchmark-results.md").write_text(
        _markdown(rows, metrics, payload["fingerprint"]),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "jobs": len(rows),
                "fingerprint": payload["fingerprint"],
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


def _write_csv(path: Path, rows: Sequence[BenchmarkRow]) -> None:
    values = [row.model_dump(mode="json") for row in rows]
    fieldnames = list(values[0]) if values else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def _markdown(
    rows: Sequence[BenchmarkRow],
    metrics: Sequence[BenchmarkMetrics],
    fingerprint: object,
) -> str:
    lines = [
        "# Frozen benchmark results",
        "",
        f"Deterministic result fingerprint: `{fingerprint}`",
        "",
        "| Rank | Company | Title | Category | Family | Eligibility | Gap | Relevance | Legacy |",
        "|---:|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        gap = "" if row.experience_gap_years is None else f"{row.experience_gap_years:.1f}"
        lines.append(
            f"| {row.relevance_rank} | {_md(row.company_name or '')} | "
            f"{_md(row.title)} | {row.category} | "
            f"{row.role_family} | {row.eligibility_status} | {gap} | "
            f"{row.relevance_score:.4f} | {row.legacy_blended_score:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Subset | N | Relevant | P@5 | P@10 | Recall@10 | NDCG@5 | NDCG@10 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics:
        lines.append(
            f"| {item.subset} | {item.evaluated_count} | {item.relevant_count} | "
            f"{item.precision_at_5:.4f} | {item.precision_at_10:.4f} | "
            f"{item.recall_at_10:.4f} | {item.ndcg_at_5:.4f} | "
            f"{item.ndcg_at_10:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _collapse_duplicate_groups(rows: Sequence[BenchmarkRow]) -> list[BenchmarkRow]:
    """Keep the highest-ranked representative of each explicit duplicate group."""

    seen: set[str] = set()
    independent: list[BenchmarkRow] = []
    for row in rows:
        group = row.duplicate_group_id
        if group is not None:
            if group in seen:
                continue
            seen.add(group)
        independent.append(row)
    return independent


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="evaluation/benchmark/v1")
    parser.add_argument("--output-dir", default="evaluation/results/benchmark-v1-run")
    parser.add_argument(
        "--labels-file",
        default=None,
        help="Optional label CSV; defaults to <benchmark-dir>/labels.csv.",
    )
    parser.add_argument(
        "--label-source",
        choices=("auto", "provisional", "human"),
        default="provisional",
    )
    parser.add_argument(
        "--collapse-duplicates",
        action="store_true",
        help="Evaluate one highest-ranked representative per duplicate group.",
    )
    parser.add_argument(
        "--allow-stage-drift",
        action="store_true",
        help=(
            "Evaluate current Stage-1/Stage-2 code after an intentional policy change "
            "instead of requiring equality with frozen stage snapshots."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
