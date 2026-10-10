"""Freeze existing persisted jobs and analyses into a local benchmark.

AWS access is read-only. Missing analyses are extracted once through the configured
LLM provider and checkpointed locally; no analysis is written back to DynamoDB.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import boto3  # type: ignore[import-untyped]

from job_intelligence.ai import build_embedding_provider, build_llm_provider, to_job_analysis
from job_intelligence.ai.prompts import JOB_REQUIREMENTS_PROMPT_VERSION
from job_intelligence.aws.dynamodb import (
    DynamoCompanyRepository,
    DynamoJobAnalysisRepository,
    DynamoJobRepository,
)
from job_intelligence.benchmark import (
    LABEL_NAMES,
    BenchmarkManifest,
    BenchmarkRow,
    FrozenAnalysisSet,
    FrozenJob,
    FrozenJobSet,
    categorize_job,
    evaluate_frozen_benchmark,
    load_frozen_benchmark,
    sha256_file,
    write_json,
)
from job_intelligence.change_detection import StoredJobState
from job_intelligence.config import EmbeddingProvider, LLMProvider, Settings
from job_intelligence.eligibility import EligibilityConfig, EligibilityEvaluator
from job_intelligence.filtering import (
    DeterministicJobFilter,
    FilteringConfig,
)
from job_intelligence.matching import MatchingConfig
from job_intelligence.models import CandidateProfile, JobAnalysis
from job_intelligence.role_policy import RoleFamilyClassifier

LABEL_FIELDS = (
    "job_id",
    "title",
    "extracted_category",
    "role_family",
    "minimum_experience_years",
    "eligibility",
    "experience_gap_years",
    "relevance_score",
    "relevance_rank",
    "provisional_label",
    "provisional_label_name",
    "provisional_notes",
    "human_label",
    "human_notes",
)


def main() -> None:
    args = _arguments()
    settings = Settings.from_environment()
    _validate_settings(settings)
    profile_path = Path(args.profile)
    output_dir = Path(args.output_dir)
    profile = CandidateProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))

    session = boto3.Session(profile_name=args.aws_profile, region_name=settings.aws_region)
    table = session.resource("dynamodb").Table(_required(settings.dynamodb_table_name))
    companies = DynamoCompanyRepository(table).list()
    company = next(
        (item for item in companies if item.name.casefold() == args.company.casefold()),
        None,
    )
    if company is None:
        raise LookupError(f"company not found: {args.company}")
    states = [
        state
        for state in DynamoJobRepository(table).list()
        if state.job.company_id == company.id and state.versions
    ]
    selected = _select_balanced(states, args.sample_size)
    if not selected:
        raise LookupError(f"no persisted jobs found for {company.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "candidate_profile.json", profile.model_dump(mode="json"))

    analyses, source_counts = _freeze_analyses(
        selected,
        output_dir,
        settings,
        DynamoJobAnalysisRepository(table),
    )
    write_json(
        output_dir / "job_analyses.json",
        FrozenAnalysisSet(analyses=analyses).model_dump(mode="json"),
    )

    filtering_config = FilteringConfig.from_settings(settings)
    eligibility_config = EligibilityConfig.from_settings(settings)
    frozen_jobs = _build_frozen_jobs(
        selected,
        analyses,
        profile,
        filtering_config,
        eligibility_config,
    )
    write_json(output_dir / "jobs.json", frozen_jobs.model_dump(mode="json"))
    manifest = BenchmarkManifest(
        benchmark_version=args.benchmark_version,
        created_at=datetime.now(timezone.utc),
        company_name=company.name,
        job_count=len(selected),
        profile_version=args.profile_version or profile_path.stem,
        profile_sha256=sha256_file(output_dir / "candidate_profile.json"),
        file_sha256={
            name: sha256_file(output_dir / name)
            for name in ("jobs.json", "job_analyses.json", "candidate_profile.json")
        },
        analysis_provider="bedrock_mantle",
        analysis_model_id=settings.mantle_model_id,
        analysis_prompt_version=JOB_REQUIREMENTS_PROMPT_VERSION,
        analyses_reused_local=source_counts["local"],
        analyses_reused_repository=source_counts["repository"],
        analyses_extracted=source_counts["extracted"],
        embedding_provider=settings.embedding_provider.value,
        embedding_model_id=settings.local_embedding_model_id,
        embedding_device=settings.local_embedding_device,
        matching_config=MatchingConfig.from_settings(settings),
        filtering_config=filtering_config,
        eligibility_config=eligibility_config,
    )
    write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))

    benchmark = load_frozen_benchmark(output_dir)
    rows = evaluate_frozen_benchmark(
        benchmark,
        build_embedding_provider(settings),
    )
    _write_labels(
        output_dir,
        rows,
        analyses,
        Path(args.provisional_labels) if args.provisional_labels else None,
    )
    _write_readme(output_dir, manifest)
    print(
        json.dumps(
            {
                "benchmark": str(output_dir),
                "jobs": len(selected),
                "analyses": source_counts,
                "profile_sha256": manifest.profile_sha256,
            },
            sort_keys=True,
        )
    )


def _freeze_analyses(
    states: Sequence[StoredJobState],
    output_dir: Path,
    settings: Settings,
    repository: DynamoJobAnalysisRepository,
) -> tuple[list[JobAnalysis], dict[str, int]]:
    checkpoint = _load_analysis_checkpoint(output_dir / "job_analyses.json")
    llm = None
    frozen: list[JobAnalysis] = []
    counts = {"local": 0, "repository": 0, "extracted": 0}
    for index, state in enumerate(states, 1):
        version = _latest_version(state)
        existing = checkpoint.get(state.job.id)
        if existing is not None and _analysis_matches(existing, version.id, settings):
            analysis = existing
            source = "local"
        else:
            cached = repository.get(
                state.job.id,
                version.id,
                settings.mantle_model_id,
                JOB_REQUIREMENTS_PROMPT_VERSION,
            )
            if cached is not None:
                analysis = cached
                source = "repository"
            else:
                if llm is None:
                    llm = build_llm_provider(settings)
                result = llm.extract_job_requirements(state.job.description)
                analysis = to_job_analysis(
                    result,
                    job_id=state.job.id,
                    job_version_id=version.id,
                )
                source = "extracted"
        frozen.append(analysis)
        counts[source] += 1
        write_json(
            output_dir / "job_analyses.json",
            FrozenAnalysisSet(analyses=frozen).model_dump(mode="json"),
        )
        print(
            f"analysis {index}/{len(states)} job_id={state.job.id} source={source}",
            flush=True,
        )
    return frozen, counts


def _build_frozen_jobs(
    states: Sequence[StoredJobState],
    analyses: Sequence[JobAnalysis],
    profile: CandidateProfile,
    filtering_config: FilteringConfig,
    eligibility_config: EligibilityConfig,
) -> FrozenJobSet:
    role_policy = filtering_config.role_policy
    if role_policy is None:
        raise ValueError("benchmark filtering configuration requires a role policy")
    classifier = RoleFamilyClassifier(role_policy)
    job_filter = DeterministicJobFilter(filtering_config)
    eligibility = EligibilityEvaluator(eligibility_config)
    analyses_by_id = {analysis.job_id: analysis for analysis in analyses}
    jobs: list[FrozenJob] = []
    for state in states:
        family = classifier.classify(state.job.title)
        analysis = analyses_by_id[state.job.id]
        jobs.append(
            FrozenJob(
                job=state.job,
                job_version_id=_latest_version(state).id,
                category=categorize_job(state.job.title),
                role_family=family,
                stage1_filter=job_filter.evaluate(state.job, profile),
                eligibility=eligibility.evaluate(
                    profile,
                    state.job,
                    analysis,
                    role_family_class=family.classification,
                ),
            )
        )
    return FrozenJobSet(jobs=jobs)


def _load_analysis_checkpoint(path: Path) -> dict[Any, JobAnalysis]:
    if not path.exists():
        return {}
    try:
        payload = FrozenAnalysisSet.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return {analysis.job_id: analysis for analysis in payload.analyses}


def _analysis_matches(analysis: JobAnalysis, version_id: Any, settings: Settings) -> bool:
    return (
        analysis.job_version_id == version_id
        and analysis.model_id == settings.mantle_model_id
        and analysis.prompt_version == JOB_REQUIREMENTS_PROMPT_VERSION
    )


def _select_balanced(
    states: Sequence[StoredJobState],
    sample_size: int,
) -> list[StoredJobState]:
    ordered = sorted(states, key=lambda state: (state.job.title.casefold(), str(state.job.id)))
    if len(ordered) <= sample_size:
        return ordered
    groups: dict[str, deque[StoredJobState]] = defaultdict(deque)
    for state in ordered:
        groups[categorize_job(state.job.title)].append(state)
    selected: list[StoredJobState] = []
    while len(selected) < sample_size:
        added = False
        for category in sorted(groups):
            if groups[category] and len(selected) < sample_size:
                selected.append(groups[category].popleft())
                added = True
        if not added:
            break
    return selected


def _write_labels(
    output_dir: Path,
    rows: Sequence[BenchmarkRow],
    analyses: Sequence[JobAnalysis],
    provisional_path: Path | None,
) -> None:
    existing_human = _existing_human_labels(output_dir / "labels.csv")
    provisional = _provisional_labels(provisional_path)
    analyses_by_id = {analysis.job_id: analysis for analysis in analyses}
    records: list[dict[str, object]] = []
    for row in rows:
        suggested = provisional.get(str(row.job_id))
        if suggested is None:
            label, notes = _suggest_label(row.title, row.category)
        else:
            label, notes = suggested
        human_label, human_notes = existing_human.get(str(row.job_id), ("", ""))
        records.append(
            {
                "job_id": str(row.job_id),
                "title": row.title,
                "extracted_category": row.category,
                "role_family": row.role_family,
                "minimum_experience_years": analyses_by_id[row.job_id].min_experience_years,
                "eligibility": row.eligibility_status,
                "experience_gap_years": row.experience_gap_years,
                "relevance_score": f"{row.relevance_score:.12f}",
                "relevance_rank": row.relevance_rank,
                "provisional_label": label,
                "provisional_label_name": LABEL_NAMES[label],
                "provisional_notes": notes,
                "human_label": human_label,
                "human_notes": human_notes,
            }
        )
    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Benchmark labels",
        "",
        "Labels: `0=irrelevant`, `1=adjacent`, `2=relevant_stretch`, "
        "`3=strong_target`. Provisional labels are suggestions only. Fill in "
        "`human_label` and `human_notes` in `labels.csv`; do not edit frozen inputs.",
        "",
        "| Rank | Title | Family | Eligibility | Relevance | Provisional | Human | Notes |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            "| {relevance_rank} | {title} | {role_family} | {eligibility} | "
            "{relevance_score} | {provisional_label} | {human_label} | {human_notes} |".format(
                **{key: _markdown(value) for key, value in record.items()}
            )
        )
    (output_dir / "labels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _provisional_labels(path: Path | None) -> dict[str, tuple[int, str]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["job_id"]): (
            int(item["provisional_label"]),
            str(item.get("rationale") or item.get("provisional_notes") or ""),
        )
        for item in payload
    }


def _existing_human_labels(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {
            row["job_id"]: (
                (row.get("human_label") or "").strip(),
                (row.get("human_notes") or "").strip(),
            )
            for row in csv.DictReader(file)
        }


def _suggest_label(title: str, category: str) -> tuple[int, str]:
    normalized = title.casefold()
    if "data scientist" in normalized or "machine learning" in normalized:
        return 2, "Provisional direct target-family label; human review required."
    if category != "clearly_unsuitable_or_other":
        return 1, "Provisional adjacent-family label; human review required."
    return 0, "Provisional non-target label; human review required."


def _write_readme(output_dir: Path, manifest: BenchmarkManifest) -> None:
    text = f"""# JobRadar frozen benchmark {manifest.benchmark_version}

This directory contains {manifest.job_count} immutable job/profile/analysis inputs and
editable labels. `manifest.json` records SHA-256 hashes and the exact configurations.

Run offline evaluation:

```powershell
python scripts/run_frozen_benchmark.py --benchmark-dir {output_dir.as_posix()}
```

Edit only `human_label` and `human_notes` in `labels.csv`. Provisional labels are not
ground truth. The runner uses human labels when present and otherwise reports the
explicit provisional baseline.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _latest_version(state: StoredJobState) -> Any:
    return max(state.versions, key=lambda version: version.version_number)


def _validate_settings(settings: Settings) -> None:
    if settings.llm_provider != LLMProvider.BEDROCK_MANTLE:
        raise ValueError("benchmark freezing requires LLM provider bedrock_mantle")
    if settings.embedding_provider != EmbeddingProvider.HUGGINGFACE_LOCAL:
        raise ValueError("benchmark freezing requires embedding provider huggingface_local")
    _required(settings.aws_region)
    _required(settings.dynamodb_table_name)
    EligibilityConfig.from_settings(settings)


def _required(value: str | None) -> str:
    if value is None:
        raise ValueError("required benchmark environment configuration is missing")
    return value


def _markdown(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aws-profile", default=None)
    parser.add_argument("--company", default="YipitData")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument(
        "--profile", default="evaluation/benchmark/v2/candidate_profile.json"
    )
    parser.add_argument("--profile-version", default=None)
    parser.add_argument("--benchmark-version", default="v1")
    parser.add_argument("--output-dir", default="evaluation/benchmark/v1")
    parser.add_argument(
        "--provisional-labels",
        default="evaluation/results/profile-impact/profile-labels.json",
    )
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    return args


if __name__ == "__main__":
    main()
