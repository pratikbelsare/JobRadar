"""Evaluate normalized role titles and soft preferred locations on frozen Benchmark v2.

This script never changes the benchmark, filtering configuration, candidate profile, or
production classifier. It uses frozen jobs and human labels to produce a counterfactual
Stage-1 report without embeddings, LLMs, AWS, or ranking recalculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID

from job_intelligence.benchmark import FrozenJob, load_frozen_benchmark, write_json
from job_intelligence.filtering import DeterministicJobFilter
from job_intelligence.models import CandidateProfile, Job, WorkMode
from job_intelligence.normalization import normalize_text
from job_intelligence.role_policy import RoleFamilyClass
from job_intelligence.role_title_normalization import (
    ExplainableRoleClassification,
    NormalizedRoleFamilyClassifier,
)

FAMILIES = ("core", "adjacent", "unknown", "exclude")


def main() -> None:
    args = _arguments()
    benchmark = load_frozen_benchmark(Path(args.benchmark_dir))
    labels = _load_labels(Path(args.labels_file))
    if set(labels) != {item.job.id for item in benchmark.jobs}:
        raise ValueError("human-label file must contain every frozen benchmark job exactly once")
    if any(label["human_label"] is None for label in labels.values()):
        raise ValueError("counterfactual evaluation requires completed human labels")
    policy = benchmark.manifest.filtering_config.role_policy
    if policy is None:
        raise ValueError("frozen benchmark has no role-family policy")
    classifier = NormalizedRoleFamilyClassifier(policy)
    production_filter = DeterministicJobFilter(benchmark.manifest.filtering_config)

    rows: list[dict[str, Any]] = []
    for item in benchmark.jobs:
        label = labels[item.job.id]
        proposed = classifier.classify(item.job.title)
        frozen_family = _family(item)
        live_original_family = _classification(proposed.original_match.classification)
        if live_original_family != frozen_family:
            raise ValueError(f"current role policy drifted for frozen job {item.job.id}")
        role_retained = _role_retained(proposed, policy)
        preferred_location_match = _preferred_location_match(
            item.job,
            benchmark.profile,
            benchmark.manifest.filtering_config.location_aliases,
        )
        hard_blocks = _hard_constraint_blocks(item.job, benchmark.profile)
        production_result = production_filter.evaluate(item.job, benchmark.profile)
        rows.append(
            {
                "job_id": str(item.job.id),
                "current_rank": label["current_rank"],
                "company": item.company_name or "",
                "title": item.job.title,
                "human_label": label["human_label"],
                "relevance_score": label["relevance_score"],
                "current_family": frozen_family,
                "frozen_stage1_passed": item.stage1_filter.passed,
                "proposed_family": _classification(proposed.match.classification),
                "matched_family": proposed.match.family_id or "",
                "matched_rule_type": (
                    proposed.match.matched_by.value if proposed.match.matched_by else ""
                ),
                "matched_rule": proposed.match.matched_term or "",
                "original_title": proposed.title.original_title,
                "normalized_title": proposed.title.normalized_title,
                "base_title": proposed.title.base_title,
                "seniority": ";".join(value.value for value in proposed.title.seniority),
                "level": proposed.title.level or "",
                "specialization": ";".join(proposed.title.specializations),
                "annotations": ";".join(proposed.title.annotations),
                "transformations": ";".join(proposed.title.transformations),
                "used_normalized_base": proposed.used_normalized_base,
                "classification_reason": proposed.reason,
                "unknown_failure_patterns": ";".join(_unknown_patterns(proposed)),
                "role_retained": role_retained,
                "preferred_location_match": preferred_location_match,
                "hard_block_reasons": ";".join(hard_blocks),
                "production_stage1_passed": production_result.passed,
                "production_stage1_outcome": production_result.outcome.value,
                "retained_with_current_location_gate": (
                    role_retained and preferred_location_match and not hard_blocks
                ),
                "retained_with_soft_preference": role_retained and not hard_blocks,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(rows)
    payload = {
        "benchmark_version": benchmark.manifest.benchmark_version,
        "benchmark_semantic_fingerprint": benchmark.manifest.semantic_fingerprint,
        "labels_file": str(Path(args.labels_file)),
        "scope": "counterfactual_only_production_behavior_unchanged",
        "summary": summary,
        "results": rows,
    }
    write_json(output_dir / "stage1-counterfactual.json", payload)
    _write_csv(output_dir / "stage1-counterfactual.csv", rows)
    _write_unknown_report(output_dir / "unknown-title-diagnosis.md", rows)
    _write_retained_ranking(output_dir / "retained-ranking.md", rows)
    (output_dir / "summary.md").write_text(
        _summary_markdown(summary, rows),
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    before = Counter(str(row["current_family"]) for row in rows)
    after = Counter(str(row["proposed_family"]) for row in rows)
    by_label_before = _family_by_label(rows, "current_family")
    by_label_after = _family_by_label(rows, "proposed_family")
    label3 = [row for row in rows if row["human_label"] == 3]
    relevant = [row for row in rows if int(row["human_label"]) >= 2]
    irrelevant = [row for row in rows if row["human_label"] == 0]
    role_retained = [row for row in rows if row["role_retained"]]
    retained_current_location = [
        row for row in rows if row["retained_with_current_location_gate"]
    ]
    retained_soft_location = [
        row for row in rows if row["retained_with_soft_preference"]
    ]
    unknown_rows = [row for row in rows if row["current_family"] == "unknown"]
    pattern_counts: Counter[str] = Counter()
    for row in unknown_rows:
        pattern_counts.update(
            pattern for pattern in str(row["unknown_failure_patterns"]).split(";") if pattern
        )
    return {
        "job_count": len(rows),
        "before_family_distribution": {family: before[family] for family in FAMILIES},
        "after_family_distribution": {family: after[family] for family in FAMILIES},
        "before_by_human_label": by_label_before,
        "after_by_human_label": by_label_after,
        "current_unknown_count": len(unknown_rows),
        "unknown_resolved_count": sum(
            row["proposed_family"] != "unknown" for row in unknown_rows
        ),
        "unknown_remaining_count": sum(
            row["proposed_family"] == "unknown" for row in unknown_rows
        ),
        "unknown_failure_pattern_counts": dict(sorted(pattern_counts.items())),
        "human_label_3_role_recall_before": _recall(label3, "current_family"),
        "human_label_3_role_recall_after": _retained_rate(label3),
        "human_label_gte_2_role_recall_before": _recall(relevant, "current_family"),
        "human_label_gte_2_role_recall_after": _retained_rate(relevant),
        "human_label_0_retention_before": _recall(irrelevant, "current_family"),
        "human_label_0_retention_after": _retained_rate(irrelevant),
        "role_eligible_after_normalization": len(role_retained),
        "production_stage1_retained": sum(
            bool(row["production_stage1_passed"]) for row in rows
        ),
        "production_stage1_rejected": sum(
            not bool(row["production_stage1_passed"]) for row in rows
        ),
        "jobs_reaching_stage2_before": sum(
            bool(row["frozen_stage1_passed"]) for row in rows
        ),
        "jobs_reaching_stage2_after": sum(
            bool(row["production_stage1_passed"]) for row in rows
        ),
        "retained_with_current_preferred_location_gate": len(retained_current_location),
        "retained_with_preferred_location_as_soft_preference": len(retained_soft_location),
        "role_eligible_but_preferred_location_mismatch": sum(
            row["role_retained"] and not row["preferred_location_match"]
            for row in rows
        ),
        "hard_constraint_blocks": sum(bool(row["hard_block_reasons"]) for row in rows),
    }


def _family_by_label(
    rows: Sequence[dict[str, Any]],
    family_field: str,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for label in range(4):
        counts = Counter(
            str(row[family_field]) for row in rows if row["human_label"] == label
        )
        result[str(label)] = {family: counts[family] for family in FAMILIES}
    return result


def _recall(rows: Sequence[dict[str, Any]], family_field: str) -> float:
    if not rows:
        return 0.0
    retained = sum(row[family_field] in {"core", "adjacent"} for row in rows)
    return retained / len(rows)


def _retained_rate(rows: Sequence[dict[str, Any]]) -> float:
    return (
        sum(bool(row["production_stage1_passed"]) for row in rows) / len(rows)
        if rows
        else 0.0
    )


def _role_retained(classification: ExplainableRoleClassification, policy: Any) -> bool:
    match = classification.match
    if match.classification not in {RoleFamilyClass.CORE, RoleFamilyClass.ADJACENT}:
        return False
    if match.family_id is None:
        return False
    return bool(policy.is_eligible(policy.family_by_id(match.family_id)))


def _preferred_location_match(
    job: Job,
    profile: CandidateProfile,
    aliases: dict[str, list[str]],
) -> bool:
    preferred = profile.preferred_locations
    preferred_modes = profile.preferences.preferred_work_modes
    if not preferred and not preferred_modes:
        return True
    location = _key(job.location or "")
    for configured in preferred:
        terms = [configured, *aliases.get(_key(configured), [])]
        if any(_contains_term(location, _key(term)) for term in terms):
            return True
    if job.work_mode in preferred_modes:
        return True
    if WorkMode.REMOTE in preferred_modes and (
        job.work_mode == WorkMode.REMOTE or _contains_term(location, "remote")
    ):
        return True
    return job.location is None and job.work_mode == WorkMode.UNKNOWN


def _hard_constraint_blocks(job: Job, profile: CandidateProfile) -> list[str]:
    hard = profile.hard_constraints
    location = _key(job.location or "")
    reasons: list[str] = []
    if any(_contains_term(location, _key(item)) for item in hard.excluded_locations):
        reasons.append("excluded_location")
    if hard.remote_only and not (
        job.work_mode == WorkMode.REMOTE or _contains_term(location, "remote")
    ):
        reasons.append("remote_only_mismatch")
    if job.employment_type in hard.excluded_employment_types:
        reasons.append("excluded_employment_type")
    searchable = _key(" ".join(filter(None, [job.title, job.location, job.description])))
    if any(_contains_term(searchable, _key(term)) for term in hard.excluded_terms):
        reasons.append("excluded_term")
    return reasons


def _unknown_patterns(result: ExplainableRoleClassification) -> list[str]:
    if result.original_match.classification is not None:
        return []
    title = result.title
    patterns: list[str] = []
    transformations = set(title.transformations)
    if "mixed_seniority_prefix_removed" in transformations:
        patterns.append("mixed_seniority_prefix")
    elif "seniority_prefix_removed" in transformations:
        patterns.append("seniority_prefix")
    if "seniority_suffix_removed" in transformations:
        patterns.append("seniority_suffix")
    if "level_suffix_removed" in transformations:
        patterns.append("level_suffix")
    if "specialization_suffix_removed" in transformations:
        patterns.append(
            "combined_title"
            if re.search(r"\s[/|]\s", title.normalized_title)
            else "specialization_suffix"
        )
    if title.annotations:
        patterns.append(
            "multilingual_parenthetical"
            if any(not character.isascii() for value in title.annotations for character in value)
            else "parenthetical_annotation"
        )
    if re.match(r"^sr\.", title.normalized_title, flags=re.IGNORECASE):
        patterns.append("seniority_abbreviation")
    if re.search(r"\bsde\b", title.normalized_title, flags=re.IGNORECASE):
        patterns.append("role_abbreviation")
    if result.match.classification is None:
        patterns.append("base_role_not_in_configured_taxonomy")
        if re.search(r"\b(?:ai|ml|data|analytics)\b", title.normalized_title, re.IGNORECASE):
            patterns.append("keyword_present_but_unsupported_function")
        if not transformations:
            patterns.append("no_structural_variant_to_normalize")
    else:
        patterns.append("resolved_by_structural_normalization")
    return _deduplicate(patterns)


def _classification(value: RoleFamilyClass | None) -> str:
    return value.value if value is not None else "unknown"


def _family(item: FrozenJob) -> str:
    return _classification(item.role_family.classification)


def _load_labels(path: Path) -> dict[UUID, dict[str, Any]]:
    result: dict[UUID, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            job_id = UUID(row["job_id"])
            human = (row.get("human_label") or "").strip()
            result[job_id] = {
                "human_label": int(human) if human else None,
                "current_rank": int(row["current_rank"]),
                "relevance_score": float(row["relevance_score"]),
            }
    if len(result) != 84:
        raise ValueError(f"expected 84 unique labels, found {len(result)}")
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_unknown_report(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    unknown = [row for row in rows if row["current_family"] == "unknown"]
    lines = [
        "# Benchmark v2 UNKNOWN title diagnosis",
        "",
        "The table records every title classified UNKNOWN by the frozen Stage-1 policy.",
        "No relevance score or human label was used to produce the normalized title.",
        "",
        (
            "| Rank | Title | Label | Base title | Seniority | Specialization | "
            "After | Failure patterns |"
        ),
        "|---:|---|---:|---|---|---|---|---|",
    ]
    for row in unknown:
        lines.append(
            f"| {row['current_rank']} | {_md(row['title'])} | {row['human_label']} | "
            f"{_md(row['base_title'])} | {_md(row['seniority'])} | "
            f"{_md(row['specialization'])} | {row['proposed_family']} | "
            f"{_md(row['unknown_failure_patterns'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_retained_ranking(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    retained = sorted(
        (row for row in rows if row["production_stage1_passed"]),
        key=lambda row: int(row["current_rank"]),
    )
    lines = [
        "# Stage-1 retained relevance ranking",
        "",
        "Scores and ranks are copied unchanged from frozen Benchmark v2.",
        "",
        "| Relevance rank | Title | Company | Human label | Family | Score |",
        "|---:|---|---|---:|---|---:|",
    ]
    for row in retained:
        lines.append(
            f"| {row['current_rank']} | {_md(row['title'])} | {_md(row['company'])} | "
            f"{row['human_label']} | {row['proposed_family']} | "
            f"{float(row['relevance_score']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_markdown(summary: dict[str, Any], rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Stage-1 title-normalization counterfactual",
        "",
        "Production behavior is unchanged. Relevance scores are copied from the frozen",
        "human-label file and are not recalculated.",
        "",
        "## Family distribution",
        "",
        "| Family | Before | After |",
        "|---|---:|---:|",
    ]
    before = summary["before_family_distribution"]
    after = summary["after_family_distribution"]
    for family in FAMILIES:
        lines.append(f"| {family.upper()} | {before[family]} | {after[family]} |")
    lines.extend(
        [
            "",
            "## Human-label metrics",
            "",
            "| Metric | Before | After |",
            "|---|---:|---:|",
            (
                "| Label 3 role recall | "
                f"{summary['human_label_3_role_recall_before']:.4f} | "
                f"{summary['human_label_3_role_recall_after']:.4f} |"
            ),
            (
                "| Label >=2 role recall | "
                f"{summary['human_label_gte_2_role_recall_before']:.4f} | "
                f"{summary['human_label_gte_2_role_recall_after']:.4f} |"
            ),
            (
                "| Label 0 retained | "
                f"{summary['human_label_0_retention_before']:.4f} | "
                f"{summary['human_label_0_retention_after']:.4f} |"
            ),
            "",
            "## Location counterfactual",
            "",
            f"Role-eligible after normalization: {summary['role_eligible_after_normalization']}.",
            (
                "Retained with current preferred-location hard gate: "
                f"{summary['retained_with_current_preferred_location_gate']}."
            ),
            (
                "Retained with preferred location treated as a soft preference: "
                f"{summary['retained_with_preferred_location_as_soft_preference']}."
            ),
            f"Production Stage-1 retained: {summary['production_stage1_retained']}.",
            f"Production Stage-1 rejected: {summary['production_stage1_rejected']}.",
            "",
            "## Classification by human label",
            "",
            "| Label | CORE | ADJACENT | UNKNOWN | EXCLUDE |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    by_label = summary["after_by_human_label"]
    for label in range(4):
        values = by_label[str(label)]
        lines.append(
            f"| {label} | {values['core']} | {values['adjacent']} | "
            f"{values['unknown']} | {values['exclude']} |"
        )
    lines.extend(
        [
            "",
            "## Safety cases",
            "",
            "| Title | Human | Proposed family | Base title |",
            "|---|---:|---|---|",
        ]
    )
    safety_terms = (
        "Product Manager, AI Infrastructure",
        "Staff Engineer 1 – Salesforce AI",
        "Staff Security Engineer",
        "Senior Data Engineer",
        "Machine Learning Operations Engineer",
        "AI Transformation Architect",
        "AI Transformation Engineer",
        "Software Engineer",
        "Data Analyst",
        "Data Engineer",
    )
    for row in rows:
        if any(term.casefold() in str(row["title"]).casefold() for term in safety_terms):
            lines.append(
                f"| {_md(row['title'])} | {row['human_label']} | "
                f"{row['proposed_family']} | {_md(row['base_title'])} |"
            )
    return "\n".join(lines) + "\n"


def _key(value: str) -> str:
    return normalize_text(value).casefold()


def _contains_term(text: str, term: str) -> bool:
    return bool(term) and re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _deduplicate(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _md(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="evaluation/benchmark/v2")
    parser.add_argument(
        "--labels-file",
        default="evaluation/benchmark/v2/labels_v2_filled.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation/results/benchmark-v2-stage1-counterfactual",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
