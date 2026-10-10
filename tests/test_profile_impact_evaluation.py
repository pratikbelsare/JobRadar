from uuid import UUID

from scripts.evaluate_profile_impact import (
    _ndcg,
    _precision,
    build_labels,
    provisional_label,
    select_reference_states,
)


def test_reference_selection_preserves_existing_evaluation_order() -> None:
    first = type("State", (), {"job": type("Job", (), {"id": UUID(int=1)})()})()
    second = type("State", (), {"job": type("Job", (), {"id": UUID(int=2)})()})()

    selected = select_reference_states([second, first], [str(first.job.id), str(second.job.id)])

    assert selected == [first, second]


def test_provisional_labels_do_not_depend_on_scores() -> None:
    label, rationale = provisional_label(
        {
            "title": "Data Scientist, Corporate",
            "category": "data_science",
            "minimum_experience_years": 4.0,
            "after_final": 0.01,
        }
    )

    assert label == 2
    assert "experience" in rationale


def test_label_artifact_keeps_human_label_editable() -> None:
    labels = build_labels(
        [
            {
                "job_id": "job-1",
                "title": "Account Executive",
                "category": "unsuitable_sales",
                "minimum_experience_years": 2.0,
            }
        ]
    )

    assert labels[0]["provisional_label"] == 0
    assert labels[0]["human_label"] is None


def test_metrics_use_relevance_labels_and_rank_order() -> None:
    rows = [
        {"job_id": "a"},
        {"job_id": "b"},
        {"job_id": "c"},
    ]
    labels = {"a": 3, "b": 0, "c": 2}

    assert _precision(rows, labels, 2) == 0.5
    assert _ndcg(rows, labels, 3) > 0.0
