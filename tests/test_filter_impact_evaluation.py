from scripts.evaluate_filter_impact import add_filtered_ranks, compare_metrics


def test_filtered_ranks_only_include_retained_successful_jobs() -> None:
    rows = [
        {"job_id": "a", "status": "ranked", "filtered_score": 0.4},
        {"job_id": "b", "status": "filtered_out", "filtered_score": None},
        {"job_id": "c", "status": "ranked", "filtered_score": 0.8},
    ]

    result = add_filtered_ranks(rows)

    assert result[0]["filtered_rank"] == 2
    assert result[1]["filtered_rank"] is None
    assert result[2]["filtered_rank"] == 1


def test_filter_metrics_count_removed_irrelevant_and_relevant_jobs() -> None:
    rows = [
        {
            "job_id": "a",
            "status": "ranked",
            "filter_passed": True,
            "filtered_score": 0.8,
        },
        {
            "job_id": "b",
            "status": "filtered_out",
            "filter_passed": False,
            "filtered_score": None,
        },
        {
            "job_id": "c",
            "status": "filtered_out",
            "filter_passed": False,
            "filtered_score": None,
        },
    ]
    previous = {
        "a": {"job_id": "a", "status": "ok", "after_final": 0.8},
        "b": {"job_id": "b", "status": "ok", "after_final": 0.7},
        "c": {"job_id": "c", "status": "ok", "after_final": 0.6},
    }
    labels = {
        "a": {"provisional_label": 2},
        "b": {"provisional_label": 0},
        "c": {"provisional_label": 2},
    }

    metrics = compare_metrics(rows, previous, labels)

    assert metrics["jobs_removed"] == 2
    assert metrics["jobs_retained"] == 1
    assert metrics["irrelevant_removed"] == 1
    assert metrics["relevant_or_stretch_removed"] == 1
    assert metrics["after_filtering"]["recall_relevant_retained"] == 0.5
    assert metrics["after_filtering"]["ndcg_at_5"] == 1.0
