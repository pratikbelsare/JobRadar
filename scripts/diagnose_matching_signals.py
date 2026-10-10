"""Read-only diagnosis of skill aliases and missing-signal score sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import boto3

from job_intelligence.ai import AIProviderError, build_ai_providers, to_job_analysis
from job_intelligence.aws.dynamodb import DynamoJobRepository
from job_intelligence.config import Settings
from job_intelligence.matching import HybridJobMatcher, MatchingConfig
from job_intelligence.models import CandidateProfile, JobAnalysis
from job_intelligence.normalization import normalize_text

TARGET_TITLES = {
    "Data Scientist, Corporate",
    "Data Product Analyst, Corporate",
    "Data QA Associate",
    "Investor Research Analyst",
    "Product Manager, AI",
}

# Deliberately conservative counterfactual aliases. These are not production configuration.
HYPOTHETICAL_SKILL_TAXONOMY = {
    "scikit-learn": {"sklearn", "scikit learn"},
    "generative ai": {"genai", "generative artificial intelligence"},
    "large language model": {"llm", "llms", "large language models"},
    "amazon s3": {"s3", "aws s3", "amazon s3"},
    "amazon ec2": {"ec2", "amazon ec2"},
    "azure openai": {"openai on azure", "azure openai"},
    "xgboost": {"xgboost", "xg boost", "xg-boost"},
    "lightgbm": {"lightgbm", "light gbm", "light-gbm"},
    "retrieval augmented generation": {
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
    },
}


def canonical_skill(value: str) -> str:
    key = normalize_text(value).casefold()
    for canonical, aliases in HYPOTHETICAL_SKILL_TAXONOMY.items():
        if key == canonical or key in aliases:
            return canonical
    return key


def hypothetical_skill_score(
    profile: CandidateProfile,
    analysis: JobAnalysis,
    *,
    required_weight: float = 0.7,
    preferred_weight: float = 0.3,
) -> float:
    candidate_skills = {canonical_skill(skill) for skill in profile.skills}
    evidenced_skills = {canonical_skill(skill) for skill in profile.skill_evidence}
    for experience in profile.work_experience:
        evidenced_skills.update(canonical_skill(skill) for skill in experience.skills)
    for project in profile.projects:
        evidenced_skills.update(canonical_skill(skill) for skill in project.skills)

    required = [canonical_skill(skill) for skill in analysis.required_skills]
    preferred = [canonical_skill(skill) for skill in analysis.preferred_skills]
    required_score = _skill_average(required, candidate_skills, evidenced_skills)
    preferred_score = _skill_average(preferred, candidate_skills, evidenced_skills)
    if required and preferred:
        return required_weight * required_score + preferred_weight * preferred_score
    if required:
        return required_score
    if preferred:
        return preferred_score
    return 0.5


def _skill_average(skills: list[str], candidate: set[str], evidence: set[str]) -> float:
    if not skills:
        return 0.5
    values = [1.0 if skill in evidence else 0.75 if skill in candidate else 0.0 for skill in skills]
    return sum(values) / len(values)


def missing_components(match: Any) -> set[str]:
    return set(match.evidence["aggregation"]["excluded_components"])


def counterfactual_scores(match: Any, missing: set[str]) -> dict[str, float]:
    scores = {
        "role": match.role_match,
        "skill": match.skill_match,
        "experience": match.experience_match,
        "responsibility": match.responsibility_match,
        "domain": match.domain_match,
        "location": match.location_match,
        "preference": match.preference_match,
    }
    weights = match.evidence["weights"]
    current = sum(scores[name] * weights[name] for name in scores)
    excluded_weight = sum(weights[name] for name in missing)
    excluded_numerator = sum(
        scores[name] * weights[name] for name in scores if name not in missing
    )
    zero_missing = current - sum(0.5 * weights[name] for name in missing)
    return {
        "previous_neutral_0_5": current,
        "normalized_available_weighted_mean": match.final_score,
        "exclude_missing_denominator": (
            excluded_numerator / (1.0 - excluded_weight)
            if excluded_weight < 1.0
            else 0.0
        ),
        "missing_as_zero": zero_missing,
    }


def evaluate(
    settings: Settings,
    profile: CandidateProfile,
    output_dir: Path,
) -> dict[str, Any]:
    region = _required(settings.aws_region, "JOB_INTELLIGENCE_AWS_REGION")
    table_name = _required(
        settings.dynamodb_table_name,
        "JOB_INTELLIGENCE_DYNAMODB_TABLE_NAME",
    )
    table = boto3.resource("dynamodb", region_name=region).Table(table_name)
    states = [
        state
        for state in DynamoJobRepository(table).list()
        if state.job.title in TARGET_TITLES
    ]
    providers = build_ai_providers(settings)
    matching_config = MatchingConfig.from_settings(settings)
    matcher = HybridJobMatcher(matching_config, embedding_provider=providers.embeddings)
    rows: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda value: value.job.title):
        version = max(state.versions, key=lambda value: value.version_number, default=None)
        if version is None:
            continue
        try:
            analysis_result = providers.llm.extract_job_requirements(state.job.description)
            analysis = to_job_analysis(
                analysis_result,
                job_id=state.job.id,
                job_version_id=version.id,
            )
            match = matcher.match(state.job, profile, analysis=analysis)
            missing = missing_components(match)
            rows.append(
                {
                    "title": state.job.title,
                    "required_skills": analysis.required_skills,
                    "preferred_skills": analysis.preferred_skills,
                    "current_skill_score": match.skill_match,
                    "hypothetical_canonical_skill_score": hypothetical_skill_score(
                        profile,
                        analysis,
                        required_weight=matching_config.required_skill_weight,
                        preferred_weight=matching_config.preferred_skill_weight,
                    ),
                    "domain_score": match.domain_match,
                    "domain_evidence": match.evidence["components"]["domain"],
                    "responsibility_score": match.responsibility_match,
                    "responsibility_evidence": match.evidence["components"]["responsibility"],
                    "final_score": match.final_score,
                    "aggregation": match.evidence["aggregation"],
                    "missing_components": sorted(missing),
                    "counterfactual_final_scores": counterfactual_scores(match, missing),
                }
            )
        except (AIProviderError, ValueError) as exc:
            rows.append({"title": state.job.title, "error": str(exc)[:240]})

    result = {
        "taxonomy": {
            canonical: sorted(aliases)
            for canonical, aliases in HYPOTHETICAL_SKILL_TAXONOMY.items()
        },
        "results": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "matching-signals.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown(result, output_dir / "matching-signals.md")
    return result


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Missing-Signal Normalization Evaluation",
        "",
        "Read-only evaluation of available-signal denominator normalization.",
        "",
        "| Job | Current skill | Canonical skill | Domain | Responsibility | Current final | "
        "Previous neutral 0.5 | Normalized final | Exclude denominator | Missing as 0 | Missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["results"]:
        if "error" in row:
            lines.append(
                f"| {row['title']} | error | error | - | - | - | - | - | - | "
                f"{row['error']} |"
            )
            continue
        scores = row["counterfactual_final_scores"]
        lines.append(
            f"| {row['title']} | {row['current_skill_score']:.3f} | "
            f"{row['hypothetical_canonical_skill_score']:.3f} | {row['domain_score']:.3f} | "
            f"{row['responsibility_score']:.3f} | {row['final_score']:.3f} | "
            f"{scores['previous_neutral_0_5']:.3f} | "
            f"{scores['normalized_available_weighted_mean']:.3f} | "
            f"{scores['exclude_missing_denominator']:.3f} | "
            f"{scores['missing_as_zero']:.3f} | {', '.join(row['missing_components']) or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _required(value: str | None, name: str) -> str:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be configured")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="evaluation/profile/pratik-enriched.json")
    parser.add_argument("--output-dir", default="evaluation/results/matching-signals")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    settings = Settings.from_environment()
    profile = CandidateProfile.model_validate_json(
        Path(arguments.profile).read_text(encoding="utf-8")
    )
    result = evaluate(settings, profile, Path(arguments.output_dir))
    print(json.dumps({"evaluated": len(result["results"])}, indent=2))


if __name__ == "__main__":
    main()
