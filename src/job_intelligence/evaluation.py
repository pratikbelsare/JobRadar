"""Offline datasets, ranking baselines, and reproducible evaluation metrics."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import UUID

from pydantic import Field, NonNegativeInt, PositiveInt

from .ai import TextEmbeddingProvider
from .filtering import FilterResult
from .matching import HybridJobMatcher
from .models import (
    CandidateProfile,
    FeedbackLabel,
    FeedbackReason,
    Job,
    JobAnalysis,
    Model,
    NonEmptyText,
    Score,
)
from .normalization import normalize_text


class EvaluationDatasetError(RuntimeError):
    """Raised when a local evaluation dataset cannot be read or written."""


class EvaluationExample(Model):
    """One portable relevance label for a candidate-job pair."""

    candidate_profile_id: UUID
    job_id: UUID
    label: FeedbackLabel
    reason: FeedbackReason | None = None
    notes: NonEmptyText | None = None


class JsonlEvaluationDataset:
    """Store labels as one JSON object per line for easy local versioning."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> list[EvaluationExample]:
        if not self._path.exists():
            raise EvaluationDatasetError(f"Evaluation dataset does not exist: {self._path}")
        examples: list[EvaluationExample] = []
        try:
            with self._path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    try:
                        examples.append(EvaluationExample.model_validate(json.loads(line)))
                    except (json.JSONDecodeError, ValueError) as exc:
                        raise EvaluationDatasetError(
                            f"Invalid evaluation example on line {line_number}"
                        ) from exc
        except OSError as exc:
            raise EvaluationDatasetError(
                f"Could not read evaluation dataset: {self._path}"
            ) from exc
        return examples

    def save(self, examples: Iterable[EvaluationExample]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._path.open("w", encoding="utf-8", newline="\n") as file:
                for example in examples:
                    file.write(json.dumps(example.model_dump(mode="json"), sort_keys=True))
                    file.write("\n")
        except OSError as exc:
            raise EvaluationDatasetError(
                f"Could not write evaluation dataset: {self._path}"
            ) from exc


@dataclass(frozen=True)
class EvaluationCase:
    example: EvaluationExample
    job: Job
    profile: CandidateProfile
    analysis: JobAnalysis | None = None
    filter_result: FilterResult | None = None

    def __post_init__(self) -> None:
        if self.example.job_id != self.job.id:
            raise ValueError("evaluation example job_id does not match the supplied job")
        if self.example.candidate_profile_id != self.profile.id:
            raise ValueError(
                "evaluation example candidate_profile_id does not match the supplied profile"
            )


class EvaluationScorer(Protocol):
    def score(self, case: EvaluationCase) -> float:
        """Return a reproducible score where larger values rank higher."""


class KeywordBaseline:
    """Small deterministic token-overlap baseline."""

    def score(self, case: EvaluationCase) -> float:
        profile_terms = _profile_terms(case.profile)
        job_terms = _job_terms(case.job, case.analysis)
        if not profile_terms or not job_terms:
            return 0.5
        overlap = len(profile_terms & job_terms) / len(profile_terms | job_terms)
        title_overlap = _token_overlap(
            case.profile.target_roles,
            [case.job.title],
        )
        return max(0.0, min(1.0, 0.7 * overlap + 0.3 * title_overlap))


class EmbeddingBaseline:
    """Compare compact candidate and job text representations using injected embeddings."""

    def __init__(self, provider: TextEmbeddingProvider) -> None:
        self._provider = provider

    def score(self, case: EvaluationCase) -> float:
        candidate = _candidate_text(case.profile)
        job = _job_text(case.job, case.analysis)
        left = self._provider.embed_text(candidate).value.values
        right = self._provider.embed_text(job).value.values
        return _cosine_similarity(left, right)


class HybridRankingBaseline:
    """Adapter for the existing Milestone 7 matcher."""

    def __init__(self, matcher: HybridJobMatcher) -> None:
        self._matcher = matcher

    def score(self, case: EvaluationCase) -> float:
        return self._matcher.match(
            case.job,
            case.profile,
            analysis=case.analysis,
            filter_result=case.filter_result,
        ).final_score


class ApproachMetrics(Model):
    approach: NonEmptyText
    dataset_size: NonNegativeInt
    query_count: NonNegativeInt
    precision_at_5: Score
    precision_at_10: Score
    ndcg_at_10: Score


class EvaluationReport(Model):
    dataset_size: NonNegativeInt
    query_count: NonNegativeInt
    relevance_threshold: int = Field(ge=0, le=3)
    approaches: list[ApproachMetrics] = Field(default_factory=list)
    configuration: dict[str, object] = Field(default_factory=dict)


def precision_at_k(
    labels: Sequence[int | FeedbackLabel],
    k: PositiveInt,
    *,
    relevance_threshold: int = 2,
) -> float:
    """Return relevant items in the first ``k`` positions divided by ``k``.

    A result with fewer than ``k`` items uses the requested ``k`` as the denominator,
    making missing recommendations count as non-relevant.
    """

    _validate_threshold(relevance_threshold)
    if not labels:
        return 0.0
    relevant = sum(1 for label in labels[:k] if int(label) >= relevance_threshold)
    return relevant / k


def ndcg_at_k(
    labels: Sequence[int | FeedbackLabel],
    k: PositiveInt,
    *,
    relevance_threshold: int = 2,
) -> float:
    """Return normalized discounted cumulative gain using graded labels 0 through 3."""

    _validate_threshold(relevance_threshold)
    if not labels:
        return 0.0
    actual = [_gain(int(label), relevance_threshold) for label in labels[:k]]
    ideal = sorted(
        (_gain(int(label), relevance_threshold) for label in labels),
        reverse=True,
    )[:k]
    ideal_dcg = _dcg(ideal)
    return 0.0 if ideal_dcg == 0 else _dcg(actual) / ideal_dcg


def evaluate_approaches(
    cases: Sequence[EvaluationCase],
    approaches: Mapping[str, EvaluationScorer | Callable[[EvaluationCase], float]],
    *,
    relevance_threshold: int = 2,
    configuration: Mapping[str, object] | None = None,
) -> EvaluationReport:
    """Evaluate all approaches over identical candidate-specific query groups."""

    _validate_threshold(relevance_threshold)
    grouped: dict[UUID, list[EvaluationCase]] = defaultdict(list)
    for case in cases:
        grouped[case.example.candidate_profile_id].append(case)

    results: list[ApproachMetrics] = []
    for name, scorer in approaches.items():
        query_metrics: list[tuple[float, float, float]] = []
        for query_cases in grouped.values():
            ranked = sorted(
                (
                    (case, _score_case(scorer, case))
                    for case in query_cases
                ),
                key=lambda item: (-item[1], str(item[0].example.job_id)),
            )
            labels = [case.example.label for case, _ in ranked]
            query_metrics.append(
                (
                    precision_at_k(labels, 5, relevance_threshold=relevance_threshold),
                    precision_at_k(labels, 10, relevance_threshold=relevance_threshold),
                    ndcg_at_k(labels, 10, relevance_threshold=relevance_threshold),
                )
            )
        query_count = len(query_metrics)
        if query_count:
            precision_5 = sum(item[0] for item in query_metrics) / query_count
            precision_10 = sum(item[1] for item in query_metrics) / query_count
            ndcg_10 = sum(item[2] for item in query_metrics) / query_count
        else:
            precision_5 = precision_10 = ndcg_10 = 0.0
        results.append(
            ApproachMetrics(
                approach=name,
                dataset_size=len(cases),
                query_count=query_count,
                precision_at_5=precision_5,
                precision_at_10=precision_10,
                ndcg_at_10=ndcg_10,
            )
        )
    return EvaluationReport(
        dataset_size=len(cases),
        query_count=len(grouped),
        relevance_threshold=relevance_threshold,
        approaches=results,
        configuration=dict(configuration or {}),
    )


def write_report(report: EvaluationReport, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        raise EvaluationDatasetError(f"Could not write evaluation report: {destination}") from exc


def _score_case(
    scorer: EvaluationScorer | Callable[[EvaluationCase], float],
    case: EvaluationCase,
) -> float:
    value = scorer.score(case) if hasattr(scorer, "score") else scorer(case)
    if not math.isfinite(value):
        raise ValueError("evaluation scorer must return a finite score")
    return value


def _validate_threshold(threshold: int) -> None:
    if threshold < 0 or threshold > 3:
        raise ValueError("relevance_threshold must be between 0 and 3")


def _gain(label: int, threshold: int) -> float:
    return float(2**label - 1) if label >= threshold else 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))


def _profile_terms(profile: CandidateProfile) -> set[str]:
    values = [*profile.target_roles, *profile.skills, *profile.domains]
    values.extend(skill for item in profile.work_experience for skill in item.skills)
    values.extend(skill for item in profile.projects for skill in item.skills)
    return _tokens(" ".join(values))


def _job_terms(job: Job, analysis: JobAnalysis | None) -> set[str]:
    values = [job.title, job.description, job.domain or ""]
    values.extend(job.required_skills)
    values.extend(job.preferred_skills)
    values.extend(job.responsibilities)
    if analysis is not None:
        values.extend(analysis.required_skills)
        values.extend(analysis.preferred_skills)
        values.extend(analysis.responsibilities)
        values.append(analysis.domain or "")
    return _tokens(" ".join(values))


def _candidate_text(profile: CandidateProfile) -> str:
    descriptions = [
        item.description
        for item in profile.work_experience
        if item.description is not None
    ] + [item.description for item in profile.projects]
    return " ".join([*profile.target_roles, *profile.skills, *profile.domains, *descriptions])


def _job_text(job: Job, analysis: JobAnalysis | None) -> str:
    values = [job.title, job.description, *job.required_skills, *job.responsibilities]
    if analysis is not None:
        values.extend(analysis.required_skills)
        values.extend(analysis.responsibilities)
    return " ".join(values)


def _token_overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_tokens = _tokens(" ".join(left))
    right_tokens = _tokens(" ".join(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(value).casefold()))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    value = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(0.0, min(1.0, value))


__all__ = [
    "ApproachMetrics",
    "EmbeddingBaseline",
    "EvaluationCase",
    "EvaluationDatasetError",
    "EvaluationExample",
    "EvaluationReport",
    "HybridRankingBaseline",
    "JsonlEvaluationDataset",
    "KeywordBaseline",
    "evaluate_approaches",
    "ndcg_at_k",
    "precision_at_k",
    "write_report",
]
