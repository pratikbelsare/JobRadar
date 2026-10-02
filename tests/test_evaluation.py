from uuid import uuid4

import pytest

from job_intelligence.ai import AIResponseMetadata, AIResult, EmbeddingVector
from job_intelligence.evaluation import (
    EmbeddingBaseline,
    EvaluationCase,
    EvaluationDatasetError,
    EvaluationExample,
    EvaluationReport,
    HybridRankingBaseline,
    JsonlEvaluationDataset,
    KeywordBaseline,
    evaluate_approaches,
    ndcg_at_k,
    precision_at_k,
    write_report,
)
from job_intelligence.matching import HybridJobMatcher
from job_intelligence.models import CandidateProfile, Job


class FakeEmbeddingProvider:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_text(self, text: str) -> AIResult[EmbeddingVector]:
        return AIResult(
            value=EmbeddingVector(values=self.vectors[text]),
            metadata=AIResponseMetadata(provider="fake", model_id="fake-model"),
        )


def make_case(label: int, *, title: str = "Java Developer") -> EvaluationCase:
    profile = CandidateProfile(
        target_roles=["Java Developer"],
        skills=["Java"],
    )
    job = Job(
        source_job_id=str(uuid4()),
        company_id=uuid4(),
        title=title,
        description="Build Java services.",
        source_url=f"https://example.test/jobs/{uuid4()}",
    )
    example = EvaluationExample(
        candidate_profile_id=profile.id,
        job_id=job.id,
        label=label,
    )
    return EvaluationCase(example=example, job=job, profile=profile)


def test_jsonl_dataset_round_trips_and_reports_bad_input(tmp_path) -> None:
    path = tmp_path / "labels.jsonl"
    dataset = JsonlEvaluationDataset(path)
    examples = [make_case(3).example, make_case(1).example]

    dataset.save(examples)

    assert dataset.load() == examples
    path.write_text('{"label": 9}\n', encoding="utf-8")
    with pytest.raises(EvaluationDatasetError):
        dataset.load()


def test_precision_at_k_uses_configurable_relevance_threshold() -> None:
    assert precision_at_k([3, 2, 1, 0], 2) == 1.0
    assert precision_at_k([3, 2, 1, 0], 4, relevance_threshold=3) == 0.25
    assert precision_at_k([3], 5) == 0.2
    assert precision_at_k([], 5) == 0.0


def test_ndcg_handles_perfect_reversed_and_tied_rankings() -> None:
    assert ndcg_at_k([3, 2, 1, 0], 4) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 2, 3], 4) < 1.0
    assert ndcg_at_k([2, 2, 2], 3) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ndcg_at_k([3], 1, relevance_threshold=4)


def test_keyword_baseline_is_reproducible_and_role_agnostic() -> None:
    good = make_case(3, title="Java Developer")
    other = make_case(0, title="DevOps Engineer")
    scorer = KeywordBaseline()

    assert scorer.score(good) > scorer.score(other)
    assert scorer.score(good) == scorer.score(good)


def test_embedding_baseline_uses_mocked_embedding_provider() -> None:
    case = make_case(3)
    candidate_text = "Java Developer Java"
    job_text = "Java Developer Build Java services."
    provider = FakeEmbeddingProvider(
        {
            candidate_text: [1.0, 0.0],
            job_text: [1.0, 0.0],
        }
    )

    assert EmbeddingBaseline(provider).score(case) == 1.0


def test_hybrid_ranking_evaluation_compares_approaches() -> None:
    cases = [
        make_case(3),
        make_case(2, title="DevOps Engineer"),
        make_case(0, title="Sales Manager"),
    ]
    report = evaluate_approaches(
        cases,
        {
            "keyword": KeywordBaseline(),
            "hybrid": HybridRankingBaseline(HybridJobMatcher()),
        },
        configuration={"ranking_version": "hybrid-v1"},
    )

    assert isinstance(report, EvaluationReport)
    assert report.dataset_size == 3
    assert {item.approach for item in report.approaches} == {"keyword", "hybrid"}
    assert all(0.0 <= item.ndcg_at_10 <= 1.0 for item in report.approaches)
    assert report.configuration["ranking_version"] == "hybrid-v1"


def test_evaluation_groups_queries_by_candidate_and_writes_report(tmp_path) -> None:
    first = make_case(3)
    second = make_case(0, title="Sales Manager")
    report = evaluate_approaches(
        [first, second],
        {"constant": lambda _: 1.0},
    )
    report_path = tmp_path / "reports" / "latest.json"

    write_report(report, report_path)

    assert report.query_count == 2
    assert report_path.exists()
    assert '"dataset_size": 2' in report_path.read_text(encoding="utf-8")
