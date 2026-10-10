"""Compare Research Scientist policies against frozen Benchmark v2 and adversarial titles."""

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
from job_intelligence.models import JobAnalysis
from job_intelligence.research_scientist_evaluation import (
    ResearchScientistDecision,
    ResearchScientistPolicy,
    ResearchScientistPolicyEvaluator,
)
from job_intelligence.role_policy import RoleFamilyClass, RoleFamilyClassifier

NEGATIVE_TITLES = (
    "Research Scientist - Chemistry",
    "Research Scientist - Biology",
    "Research Scientist - Materials Science",
    "Research Scientist - Drug Discovery",
    "Research Scientist - Clinical Research",
    "Research Scientist - Agriculture",
    "Research Scientist - Battery Materials",
)
POSITIVE_TITLES = (
    "AI Research Scientist",
    "Machine Learning Research Scientist",
    "Research Scientist - NLP",
    "Research Scientist - Computer Vision",
    "Research Scientist - Foundation Models",
)
RELEVANCE_TERMS = (
    "artificial intelligence",
    "ai",
    "machine learning",
    "ml",
    "deep learning",
    "nlp",
    "llm",
    "language model",
    "multimodal",
    "multi-modal",
    "generative",
    "diffusion",
    "reinforcement learning",
    "model evaluation",
    "computer vision",
    "3d vision",
    "pytorch",
    "tensorflow",
    "synthetic data",
)


def main() -> None:
    args = _arguments()
    benchmark = load_frozen_benchmark(Path(args.benchmark_dir))
    labels = _load_labels(Path(args.labels_file))
    if set(labels) != {item.job.id for item in benchmark.jobs}:
        raise ValueError("labels must contain every frozen job exactly once")
    policy = benchmark.manifest.filtering_config.role_policy
    if policy is None:
        raise ValueError("benchmark has no frozen role policy")
    production = RoleFamilyClassifier(policy)
    evaluator = ResearchScientistPolicyEvaluator()

    comparisons: dict[str, Any] = {}
    benchmark_decisions: dict[str, list[dict[str, Any]]] = {}
    for candidate_policy in ResearchScientistPolicy:
        rows = _apply_policy(
            benchmark.jobs,
            labels,
            production,
            evaluator,
            candidate_policy,
        )
        adversarial = _adversarial(evaluator, candidate_policy)
        comparisons[candidate_policy.value] = _metrics(rows, adversarial)
        benchmark_decisions[candidate_policy.value] = rows

    evidence = _research_evidence(
        benchmark.jobs,
        benchmark.analyses,
        labels,
        evaluator,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "evaluation_only_production_behavior_unchanged",
        "benchmark_semantic_fingerprint": benchmark.manifest.semantic_fingerprint,
        "comparisons": comparisons,
        "benchmark_decisions": benchmark_decisions,
        "research_scientist_evidence": evidence,
    }
    write_json(output_dir / "research-scientist-policy-evaluation.json", payload)
    _write_comparison_csv(output_dir / "policy-comparison.csv", comparisons)
    _write_evidence(output_dir / "research-scientist-evidence.md", evidence)
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparisons, benchmark_decisions),
        encoding="utf-8",
    )
    print(json.dumps(comparisons, sort_keys=True))


def _apply_policy(
    jobs: Sequence[FrozenJob],
    labels: dict[UUID, int],
    production: RoleFamilyClassifier,
    evaluator: ResearchScientistPolicyEvaluator,
    policy: ResearchScientistPolicy,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in jobs:
        current = production.classify(item.job.title).classification
        decision = evaluator.evaluate(item.job.title, policy)
        projected = current
        overlay_applied = current is None and decision.applicable
        if overlay_applied:
            projected = decision.classification
        rows.append(
            {
                "job_id": str(item.job.id),
                "title": item.job.title,
                "company": item.company_name,
                "human_label": labels[item.job.id],
                "current_family": _family(current),
                "projected_family": _family(projected),
                "retained": projected in {RoleFamilyClass.CORE, RoleFamilyClass.ADJACENT},
                "overlay_applied": overlay_applied,
                "title_evidence": decision.title_evidence if decision.applicable else [],
                "matched_qualifiers": (
                    decision.matched_qualifiers if decision.applicable else []
                ),
                "reason": decision.reason if decision.applicable else "Policy not applicable.",
            }
        )
    return rows


def _metrics(
    rows: Sequence[dict[str, Any]],
    adversarial: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    labels = {label: [row for row in rows if row["human_label"] == label] for label in range(4)}
    relevant = [row for row in rows if row["human_label"] >= 2]
    distribution = Counter(str(row["projected_family"]) for row in rows)
    research_rows = [row for row in rows if row["overlay_applied"]]
    return {
        "family_distribution": {
            family: distribution[family]
            for family in ("core", "adjacent", "unknown", "exclude")
        },
        "label_3_retention": _rate(labels[3]),
        "label_2_retention": _rate(labels[2]),
        "label_gte_2_retention": _rate(relevant),
        "label_1_retention": _rate(labels[1]),
        "label_0_retention": _rate(labels[0]),
        "retained_count": sum(bool(row["retained"]) for row in rows),
        "research_scientist_overlay_count": len(research_rows),
        "research_scientist_classification": dict(
            Counter(str(row["projected_family"]) for row in research_rows)
        ),
        "adversarial_negative_core_count": sum(
            row["classification"] == "core" for row in adversarial["negative"]
        ),
        "adversarial_negative_retained_count": sum(
            row["retained"] for row in adversarial["negative"]
        ),
        "adversarial_positive_core_count": sum(
            row["classification"] == "core" for row in adversarial["positive"]
        ),
        "adversarial": adversarial,
    }


def _adversarial(
    evaluator: ResearchScientistPolicyEvaluator,
    policy: ResearchScientistPolicy,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "negative": [_decision_row(evaluator.evaluate(title, policy)) for title in NEGATIVE_TITLES],
        "positive": [_decision_row(evaluator.evaluate(title, policy)) for title in POSITIVE_TITLES],
    }


def _decision_row(decision: ResearchScientistDecision) -> dict[str, Any]:
    family = _family(decision.classification)
    return {
        "title": decision.normalized_title.original_title,
        "base_title": decision.normalized_title.base_title,
        "specializations": decision.normalized_title.specializations,
        "classification": family,
        "retained": family in {"core", "adjacent"},
        "matched_qualifiers": decision.matched_qualifiers,
        "reason": decision.reason,
    }


def _research_evidence(
    jobs: Sequence[FrozenJob],
    analyses: dict[UUID, JobAnalysis],
    labels: dict[UUID, int],
    evaluator: ResearchScientistPolicyEvaluator,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in jobs:
        decision = evaluator.evaluate(
            item.job.title,
            ResearchScientistPolicy.CONTEXT_QUALIFIED,
        )
        if not decision.applicable:
            continue
        analysis = analyses[item.job.id]
        description_values = [
            *analysis.required_skills,
            *analysis.preferred_skills,
            *analysis.responsibilities,
        ]
        records.append(
            {
                "job_id": str(item.job.id),
                "company": item.company_name,
                "original_title": item.job.title,
                "base_title": decision.normalized_title.base_title,
                "specializations": decision.normalized_title.specializations,
                "human_label": labels[item.job.id],
                "title_classification": _family(decision.classification),
                "title_qualifiers": decision.matched_qualifiers,
                "required_skills": analysis.required_skills,
                "preferred_skills": analysis.preferred_skills,
                "responsibilities": analysis.responsibilities,
                "domain": analysis.domain,
                "description_relevance_terms": _evidence_terms(description_values),
                "relevant_skill_evidence": _relevant_items(
                    [*analysis.required_skills, *analysis.preferred_skills]
                ),
                "relevant_responsibility_evidence": _relevant_items(
                    analysis.responsibilities
                ),
            }
        )
    return records


def _evidence_terms(values: Iterable[str]) -> list[str]:
    text = " ".join(values).casefold()
    return [
        term
        for term in RELEVANCE_TERMS
        if re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", text)
    ]


def _relevant_items(values: Sequence[str]) -> list[str]:
    relevant = [value for value in values if _evidence_terms([value])]
    return (relevant or list(values))[:3]


def _rate(rows: Sequence[dict[str, Any]]) -> float:
    return sum(bool(row["retained"]) for row in rows) / len(rows) if rows else 0.0


def _family(value: RoleFamilyClass | None) -> str:
    return value.value if value is not None else "unknown"


def _load_labels(path: Path) -> dict[UUID, int]:
    labels: dict[UUID, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            value = (row.get("human_label") or "").strip()
            if not value:
                raise ValueError("human labels must be complete")
            labels[UUID(row["job_id"])] = int(value)
    return labels


def _write_comparison_csv(path: Path, comparisons: dict[str, Any]) -> None:
    fields = (
        "policy",
        "label_3_retention",
        "label_2_retention",
        "label_gte_2_retention",
        "label_1_retention",
        "label_0_retention",
        "retained_count",
        "adversarial_negative_core_count",
        "adversarial_negative_retained_count",
        "adversarial_positive_core_count",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for policy, values in comparisons.items():
            writer.writerow({"policy": policy, **{field: values[field] for field in fields[1:]}})


def _write_evidence(path: Path, evidence: Sequence[dict[str, Any]]) -> None:
    lines = ["# Frozen Research Scientist evidence", ""]
    for record in evidence:
        lines.extend(
            [
                f"## {record['original_title']} — {record['company']}",
                "",
                f"- Human label: `{record['human_label']}`",
                f"- Base title: `{record['base_title']}`",
                f"- Specialization: `{', '.join(record['specializations']) or 'none'}`",
                f"- Title-only classification: `{record['title_classification']}`",
                f"- Title qualifiers: `{', '.join(record['title_qualifiers']) or 'none'}`",
                f"- Extracted domain: `{record['domain'] or 'unknown'}`",
                (
                    "- Description evidence terms: `"
                    + ", ".join(record["description_relevance_terms"])
                    + "`"
                ),
                "- Relevant skill evidence:",
                *[f"  - {_md(value)}" for value in record["relevant_skill_evidence"]],
                "- Relevant responsibility evidence:",
                *[
                    f"  - {_md(value)}"
                    for value in record["relevant_responsibility_evidence"]
                ],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_markdown(
    comparisons: dict[str, Any],
    decisions: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "# Research Scientist policy comparison",
        "",
        "Evaluation only. Production role policy is unchanged.",
        "",
        "| Policy | L3 | L2 | L>=2 | L1 | L0 | Retained | Neg CORE | Neg retained | Pos CORE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ResearchScientistPolicy:
        values = comparisons[policy.value]
        lines.append(
            f"| {policy.value} | {values['label_3_retention']:.4f} | "
            f"{values['label_2_retention']:.4f} | "
            f"{values['label_gte_2_retention']:.4f} | "
            f"{values['label_1_retention']:.4f} | "
            f"{values['label_0_retention']:.4f} | {values['retained_count']} | "
            f"{values['adversarial_negative_core_count']} | "
            f"{values['adversarial_negative_retained_count']} | "
            f"{values['adversarial_positive_core_count']} |"
        )
    lines.extend(
        [
            "",
            "## Context-qualified Benchmark v2 decisions",
            "",
            "| Title | Human | Projected family | Title evidence |",
            "|---|---:|---|---|",
        ]
    )
    for row in decisions[ResearchScientistPolicy.CONTEXT_QUALIFIED.value]:
        if row["overlay_applied"]:
            lines.append(
                f"| {_md(row['title'])} | {row['human_label']} | "
                f"{row['projected_family']} | {_md(', '.join(row['title_evidence']))} |"
            )
    return "\n".join(lines) + "\n"


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
        default="evaluation/results/benchmark-v2-research-scientist-policy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
