"""Freeze the reviewed multi-company Benchmark v2 plan.

Career-site reads and Mantle extraction happen only in this one-time freeze step.
The resulting benchmark is evaluated later with the strictly offline runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from job_intelligence.ai import build_llm_provider, to_job_analysis
from job_intelligence.ai.prompts import JOB_REQUIREMENTS_PROMPT_VERSION
from job_intelligence.benchmark import (
    V2_IMMUTABLE_BENCHMARK_FILES,
    BenchmarkManifest,
    BenchmarkRow,
    FrozenAnalysisSet,
    FrozenJob,
    FrozenJobSet,
    FrozenStage1Result,
    FrozenStage1Set,
    FrozenStage2Result,
    FrozenStage2Set,
    evaluate_frozen_benchmark,
    load_frozen_benchmark,
    semantic_benchmark_fingerprint,
    sha256_file,
    write_json,
)
from job_intelligence.benchmark_v2 import (
    BenchmarkSourcePlan,
    SelectionBucket,
    count_independent_jobs,
    duplicate_groups,
)
from job_intelligence.config import EmbeddingProvider, LLMProvider, Settings
from job_intelligence.connectors import ConnectorStatus, GreenhouseConnector, JobReference
from job_intelligence.eligibility import EligibilityEvaluator
from job_intelligence.filtering import DeterministicJobFilter
from job_intelligence.http import HttpClient
from job_intelligence.models import CandidateProfile, Job, JobAnalysis
from job_intelligence.normalization import JobNormalizer
from job_intelligence.role_policy import RoleFamilyClassifier

LABEL_FIELDS = (
    "job_id",
    "company",
    "title",
    "canonical_url",
    "location",
    "source_job_id",
    "source_updated_at",
    "extracted_category",
    "role_category",
    "role_family",
    "stage1_result",
    "stage1_reason_codes",
    "stage2_eligibility",
    "minimum_experience_years",
    "experience_gap_years",
    "relevance_score",
    "current_rank",
    "selection_bucket",
    "duplicate_group_id",
    "provisional_label",
    "human_label",
    "human_notes",
)


def main() -> None:
    args = _arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_frozen_benchmark(Path(args.reference_benchmark))
    plan = BenchmarkSourcePlan.model_validate_json(
        Path(args.source_plan).read_text(encoding="utf-8")
    )
    _validate_plan_path(plan, Path(args.source_plan), output_dir)

    profile = reference.profile
    write_json(output_dir / "candidate_profile.json", profile.model_dump(mode="json"))
    jobs, raw_descriptions = _fetch_jobs(plan, args)
    groups = duplicate_groups(jobs)
    versions = {
        job.id: uuid5(NAMESPACE_URL, f"jobradar:version:{job.id}:{job.content_hash}")
        for job in jobs
    }

    settings = _provider_settings(reference.manifest, args)
    if args.aws_profile:
        os.environ["AWS_PROFILE"] = args.aws_profile
    analyses, _ = _freeze_analyses(
        jobs,
        versions,
        output_dir / "job_analyses.json",
        settings,
    )
    write_json(
        output_dir / "job_analyses.json",
        FrozenAnalysisSet(analyses=analyses).model_dump(mode="json"),
    )

    frozen_jobs, stage1, stage2 = _build_snapshots(
        plan,
        jobs,
        versions,
        analyses,
        raw_descriptions,
        groups,
        profile,
        reference.manifest,
    )
    write_json(output_dir / "jobs.json", frozen_jobs.model_dump(mode="json"))
    write_json(output_dir / "stage1_results.json", stage1.model_dump(mode="json"))
    write_json(output_dir / "stage2_results.json", stage2.model_dump(mode="json"))

    independent_count = count_independent_jobs(groups.values())
    immutable_hashes = {
        name: sha256_file(output_dir / name) for name in V2_IMMUTABLE_BENCHMARK_FILES
    }
    manifest = BenchmarkManifest(
        schema_version="2",
        benchmark_version="v2",
        created_at=datetime.now(timezone.utc),
        company_name="Multiple official company career sites",
        companies=sorted({item.company_name for item in plan.jobs}),
        job_count=len(jobs),
        independent_job_count=independent_count,
        duplicate_group_count=len({value for value in groups.values() if value}),
        semantic_fingerprint=semantic_benchmark_fingerprint(immutable_hashes),
        profile_version=reference.manifest.profile_version,
        profile_sha256=immutable_hashes["candidate_profile.json"],
        file_sha256=immutable_hashes,
        analysis_provider="bedrock_mantle",
        analysis_model_id=settings.mantle_model_id,
        analysis_prompt_version=JOB_REQUIREMENTS_PROMPT_VERSION,
        # Checkpoints only resume this same one-time freeze. All v2 analyses were
        # originally extracted for this benchmark rather than imported from v1.
        analyses_reused_local=0,
        analyses_reused_repository=0,
        analyses_extracted=len(analyses),
        embedding_provider=reference.manifest.embedding_provider,
        embedding_model_id=reference.manifest.embedding_model_id,
        embedding_device=reference.manifest.embedding_device,
        matching_config=reference.manifest.matching_config,
        filtering_config=reference.manifest.filtering_config,
        eligibility_config=reference.manifest.eligibility_config,
    )
    write_json(output_dir / "manifest.json", manifest.model_dump(mode="json"))

    frozen = load_frozen_benchmark(output_dir)
    from job_intelligence.ai import build_embedding_provider

    rows = evaluate_frozen_benchmark(frozen, build_embedding_provider(settings))
    _write_labels(output_dir, rows, frozen_jobs.jobs, analyses)
    audit = _quality_audit(rows, frozen_jobs.jobs, analyses, manifest)
    write_json(output_dir / "audit.json", audit)
    _write_readme(output_dir, manifest, audit)
    print(json.dumps(audit, sort_keys=True), flush=True)


def _fetch_jobs(
    plan: BenchmarkSourcePlan,
    args: argparse.Namespace,
) -> tuple[list[Job], dict[UUID, str]]:
    connector = GreenhouseConnector(
        HttpClient(
            timeout_seconds=args.http_timeout,
            user_agent=args.user_agent,
            max_retries=1,
        )
    )
    normalizer = JobNormalizer()
    jobs: list[Job] = []
    descriptions: dict[UUID, str] = {}
    for index, planned in enumerate(plan.jobs, 1):
        reference = JobReference(
            company_id=planned.company_id,
            source_job_id=planned.source_job_id,
            title=planned.title,
            location=planned.location,
            source_url=planned.canonical_url,
            details_url=planned.details_url,
        )
        result = connector.get_job(reference)
        if result.status != ConnectorStatus.SUCCESS or result.job is None:
            raise RuntimeError(
                f"job fetch failed company={planned.company_name} "
                f"source_job_id={planned.source_job_id} status={result.status.value} "
                f"error={result.error}"
            )
        raw = result.job
        source_time = _source_timestamp(raw.raw_payload) or plan.created_at
        normalized = normalizer.normalize(raw, observed_at=source_time)
        job_id = uuid5(
            NAMESPACE_URL,
            f"jobradar:greenhouse:{planned.board_token}:{planned.source_job_id}",
        )
        job = Job.model_validate(
            {
                **normalized.model_dump(mode="python"),
                "id": job_id,
                "first_seen_at": source_time,
                "last_seen_at": source_time,
            }
        )
        jobs.append(job)
        descriptions[job.id] = raw.description
        print(
            f"fetched {index}/{len(plan.jobs)} company={planned.company_name} "
            f"source_job_id={planned.source_job_id}",
            flush=True,
        )
    return jobs, descriptions


def _freeze_analyses(
    jobs: Sequence[Job],
    versions: dict[UUID, UUID],
    checkpoint_path: Path,
    settings: Settings,
) -> tuple[list[JobAnalysis], dict[str, int]]:
    checkpoint = _load_analysis_checkpoint(checkpoint_path)
    provider = None
    analyses: list[JobAnalysis] = []
    counts = {"local": 0, "extracted": 0}
    for index, job in enumerate(jobs, 1):
        existing = checkpoint.get(job.id)
        if _analysis_matches(existing, versions[job.id], settings):
            assert existing is not None
            analysis = existing
            source = "local"
        else:
            if provider is None:
                provider = build_llm_provider(settings)
            result = provider.extract_job_requirements(job.description)
            analysis = to_job_analysis(
                result,
                job_id=job.id,
                job_version_id=versions[job.id],
            )
            source = "extracted"
        analyses.append(analysis)
        counts[source] += 1
        write_json(
            checkpoint_path,
            FrozenAnalysisSet(analyses=analyses).model_dump(mode="json"),
        )
        print(
            f"analysis {index}/{len(jobs)} job_id={job.id} source={source}",
            flush=True,
        )
    return analyses, counts


def _build_snapshots(
    plan: BenchmarkSourcePlan,
    jobs: Sequence[Job],
    versions: dict[UUID, UUID],
    analyses: Sequence[JobAnalysis],
    raw_descriptions: dict[UUID, str],
    groups: dict[UUID, str | None],
    profile: CandidateProfile,
    reference_manifest: BenchmarkManifest,
) -> tuple[FrozenJobSet, FrozenStage1Set, FrozenStage2Set]:
    policy = reference_manifest.filtering_config.role_policy
    if policy is None:
        raise ValueError("reference benchmark must contain the Stage-1 role policy")
    classifier = RoleFamilyClassifier(policy)
    stage1_evaluator = DeterministicJobFilter(reference_manifest.filtering_config)
    stage2_evaluator = EligibilityEvaluator(reference_manifest.eligibility_config)
    analyses_by_id = {item.job_id: item for item in analyses}
    plan_by_identity = {
        (item.company_id, item.source_job_id): item for item in plan.jobs
    }
    frozen: list[FrozenJob] = []
    first_results: list[FrozenStage1Result] = []
    second_results: list[FrozenStage2Result] = []
    for job in jobs:
        planned = plan_by_identity[(job.company_id, job.source_job_id)]
        analysis = analyses_by_id[job.id]
        family = classifier.classify(job.title)
        first = stage1_evaluator.evaluate(job, profile)
        second = stage2_evaluator.evaluate(
            profile,
            job,
            analysis,
            role_family_class=family.classification,
        )
        frozen.append(
            FrozenJob(
                job=job,
                job_version_id=versions[job.id],
                category=analysis.domain or "unknown",
                role_family=family,
                stage1_filter=first,
                eligibility=second,
                company_name=planned.company_name,
                source_updated_at=job.last_seen_at,
                role_category=planned.role_category,
                selection_bucket=planned.selection_bucket.value,
                duplicate_group_id=groups[job.id],
                raw_description=raw_descriptions[job.id],
            )
        )
        first_results.append(
            FrozenStage1Result(
                job_id=job.id,
                role_family=family,
                passed=first.passed,
                outcome=first.outcome.value,
                reason_codes=_stage1_reason_codes(first),
                result=first,
            )
        )
        second_results.append(FrozenStage2Result(job_id=job.id, result=second))
    return (
        FrozenJobSet(jobs=frozen),
        FrozenStage1Set(results=first_results),
        FrozenStage2Set(results=second_results),
    )


def _write_labels(
    directory: Path,
    rows: Sequence[BenchmarkRow],
    jobs: Sequence[FrozenJob],
    analyses: Sequence[JobAnalysis],
) -> None:
    old_labels = _existing_human_labels(directory / "labels.csv")
    jobs_by_id = {item.job.id: item for item in jobs}
    analyses_by_id = {item.job_id: item for item in analyses}
    records: list[dict[str, object]] = []
    for row in rows:
        item = jobs_by_id[row.job_id]
        analysis = analyses_by_id[row.job_id]
        human_label, human_notes = old_labels.get(str(row.job_id), ("", ""))
        records.append(
            {
                "job_id": str(row.job_id),
                "company": item.company_name,
                "title": row.title,
                "canonical_url": str(item.job.source_url),
                "location": item.job.location or "",
                "source_job_id": item.job.source_job_id,
                "source_updated_at": item.source_updated_at.isoformat()
                if item.source_updated_at
                else "",
                "extracted_category": analysis.domain or "",
                "role_category": item.role_category,
                "role_family": row.role_family,
                "stage1_result": row.stage1_outcome,
                "stage1_reason_codes": ";".join(_stage1_reason_codes(item.stage1_filter)),
                "stage2_eligibility": row.eligibility_status,
                "minimum_experience_years": analysis.min_experience_years
                if analysis.min_experience_years is not None
                else "",
                "experience_gap_years": row.experience_gap_years
                if row.experience_gap_years is not None
                else "",
                "relevance_score": f"{row.relevance_score:.12f}",
                "current_rank": row.relevance_rank,
                "selection_bucket": item.selection_bucket,
                "duplicate_group_id": item.duplicate_group_id or "",
                "provisional_label": "",
                "human_label": human_label,
                "human_notes": human_notes,
            }
        )
    with (directory / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Benchmark v2 human labels",
        "",
        "Fill `human_label` with `3=strong_target`, `2=relevant_stretch`, "
        "`1=adjacent`, or `0=irrelevant`. Selection buckets are sampling strata, "
        "not labels. Frozen inputs and scores must not be edited.",
        "",
        "| Rank | Company | Title | Family | Eligibility | Relevance | Human | Notes |",
        "|---:|---|---|---|---|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['current_rank']} | {_md(record['company'])} | "
            f"{_md(record['title'])} | {record['role_family']} | "
            f"{record['stage2_eligibility']} | {record['relevance_score']} | "
            f"{record['human_label']} | {_md(record['human_notes'])} |"
        )
    (directory / "labels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_audit(
    rows: Sequence[BenchmarkRow],
    jobs: Sequence[FrozenJob],
    analyses: Sequence[JobAnalysis],
    manifest: BenchmarkManifest,
) -> dict[str, object]:
    role_categories = Counter(item.role_category for item in jobs)
    role_families = Counter(row.role_family for row in rows)
    eligibility = Counter(row.eligibility_status for row in rows)
    return {
        "raw_job_count": len(jobs),
        "independent_job_count": manifest.independent_job_count,
        "duplicate_group_count": manifest.duplicate_group_count,
        "companies": dict(sorted(Counter(item.company_name for item in jobs).items())),
        "role_categories": {
            name: role_categories[name]
            for name in (
                "ml_engineer",
                "ai_engineer",
                "data_scientist",
                "applied_scientist",
                "genai_llm",
                "analytics_data_adjacent",
                "ambiguous_hard_negative",
                "irrelevant",
            )
        },
        "role_families": {
            name: role_families[name]
            for name in ("core", "adjacent", "unknown", "exclude")
        },
        "eligibility": {
            name: eligibility[name]
            for name in ("eligible", "stretch", "severe_stretch", "ineligible", "unknown")
        },
        "locations": dict(
            sorted(Counter(item.job.location or "unknown" for item in jobs).items())
        ),
        "seniority": dict(
            sorted(Counter(item.seniority.value for item in analyses).items())
        ),
        "selection_strata": dict(
            sorted(Counter(item.selection_bucket for item in jobs).items())
        ),
        "likely_label_3_candidates": sum(
            item.selection_bucket == SelectionBucket.TARGET_REGULAR.value for item in jobs
        ),
        "likely_label_2_candidates": sum(
            item.selection_bucket == SelectionBucket.TARGET_STRETCH.value for item in jobs
        ),
        "ambiguous_hard_negative_cases": sum(
            item.role_category == "ambiguous_hard_negative" for item in jobs
        ),
        "semantic_fingerprint": manifest.semantic_fingerprint,
    }


def _write_readme(
    directory: Path,
    manifest: BenchmarkManifest,
    audit: dict[str, object],
) -> None:
    directory_display = directory.as_posix()
    text = f"""# JobRadar frozen benchmark v2

This benchmark contains {manifest.job_count} current jobs frozen from official company
career sites. The selection strata deliberately cover target, stretch, adjacent, and
hard-negative roles; they are not human relevance labels.

The immutable files are hashed in `manifest.json`. The semantic fingerprint is
`{manifest.semantic_fingerprint}`. Raw count: {manifest.job_count}; independent count:
{manifest.independent_job_count}; duplicate groups: {manifest.duplicate_group_count}.

`job_analyses.json` was extracted once with `{manifest.analysis_provider}` /
`{manifest.analysis_model_id}` and strict schema validation. Normal evaluation never
uses Mantle, DynamoDB, SQS, or any AWS API; it uses only the frozen files and local
`{manifest.embedding_model_id}` embeddings.

Run twice offline and compare fingerprints:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.venv\\Scripts\\python.exe scripts/run_frozen_benchmark.py `
  --benchmark-dir {directory_display} `
  --output-dir evaluation/results/benchmark-v2-run-1
.venv\\Scripts\\python.exe scripts/run_frozen_benchmark.py `
  --benchmark-dir {directory_display} `
  --output-dir evaluation/results/benchmark-v2-run-2
```

To label, edit only `human_label` and `human_notes` in `labels.csv`. Use
`3=strong_target`, `2=relevant_stretch`, `1=adjacent`, `0=irrelevant`. Do not infer a
label from `selection_bucket`, current rank, role family, or eligibility.

The full pre-label quality audit is in `audit.json`.
"""
    (directory / "README.md").write_text(text, encoding="utf-8")


def _provider_settings(manifest: BenchmarkManifest, args: argparse.Namespace) -> Settings:
    if manifest.analysis_provider != LLMProvider.BEDROCK_MANTLE.value:
        raise ValueError("reference benchmark must use the Mantle analysis provider")
    if manifest.embedding_provider != EmbeddingProvider.HUGGINGFACE_LOCAL.value:
        raise ValueError("reference benchmark must use local Hugging Face embeddings")
    return Settings(
        aws_region=args.region,
        llm_provider=LLMProvider.BEDROCK_MANTLE,
        embedding_provider=EmbeddingProvider.HUGGINGFACE_LOCAL,
        mantle_model_id=manifest.analysis_model_id,
        mantle_max_tokens=args.mantle_max_tokens,
        bedrock_timeout_seconds=args.mantle_timeout,
        ai_max_attempts=args.ai_max_attempts,
        local_embedding_model_id=manifest.embedding_model_id,
        local_embedding_device=manifest.embedding_device,
    )


def _validate_plan_path(
    plan: BenchmarkSourcePlan,
    source_path: Path,
    output_dir: Path,
) -> None:
    destination = output_dir / "source_plan.json"
    if source_path.resolve() != destination.resolve():
        write_json(destination, plan.model_dump(mode="json"))


def _load_analysis_checkpoint(path: Path) -> dict[UUID, JobAnalysis]:
    if not path.exists():
        return {}
    try:
        analyses = FrozenAnalysisSet.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return {item.job_id: item for item in analyses.analyses}


def _analysis_matches(
    analysis: JobAnalysis | None,
    version_id: UUID,
    settings: Settings,
) -> bool:
    return bool(
        analysis is not None
        and analysis.job_version_id == version_id
        and analysis.provider == "bedrock_mantle"
        and analysis.model_id == settings.mantle_model_id
        and analysis.prompt_version == JOB_REQUIREMENTS_PROMPT_VERSION
    )


def _source_timestamp(payload: dict[str, Any]) -> datetime | None:
    for name in ("updated_at", "created_at", "published_at"):
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _stage1_reason_codes(result: Any) -> list[str]:
    codes: list[str] = []
    for rule in result.rules:
        code = rule.details.get("decision_code")
        if isinstance(code, str) and code and code not in codes:
            codes.append(code)
    if not codes:
        codes.append(result.outcome.value)
    return codes


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


def _md(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", default="evaluation/benchmark/v2/source_plan.json")
    parser.add_argument("--output-dir", default="evaluation/benchmark/v2")
    parser.add_argument("--reference-benchmark", default="evaluation/benchmark/v1")
    parser.add_argument("--aws-profile", default="jobradar-dev")
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--http-timeout", type=float, default=30)
    parser.add_argument("--mantle-timeout", type=float, default=180)
    parser.add_argument("--mantle-max-tokens", type=int, default=4096)
    parser.add_argument("--ai-max-attempts", type=int, default=3)
    parser.add_argument("--user-agent", default="JobRadar-Benchmark/2.0")
    return parser.parse_args()


if __name__ == "__main__":
    main()
