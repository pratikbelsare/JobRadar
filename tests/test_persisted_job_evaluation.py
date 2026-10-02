from types import SimpleNamespace
from uuid import UUID

from scripts.evaluate_persisted_jobs import category_for_title, select_sample


def state(title: str, value: int) -> SimpleNamespace:
    return SimpleNamespace(
        job=SimpleNamespace(title=title, id=UUID(int=value)),
    )


def test_evaluation_categories_cover_relevant_and_unsuitable_titles() -> None:
    assert category_for_title("Data Scientist, Corporate") == "data_science"
    assert category_for_title("Product Manager, AI") == "product_ai"
    assert category_for_title("Data Analyst") == "analytics"
    assert category_for_title("Account Executive") == "unsuitable_sales"
    assert category_for_title("Senior SDET") == "unsuitable_other"


def test_sample_selection_is_deterministic_and_round_robin() -> None:
    jobs = [
        state("Account Executive", 1),
        state("Data Scientist", 2),
        state("Data Analyst", 3),
        state("Product Manager, AI", 4),
        state("Senior SDET", 5),
    ]

    selected = select_sample(jobs, 4)

    assert [item.job.title for item in selected] == [
        "Data Scientist",
        "Product Manager, AI",
        "Data Analyst",
        "Account Executive",
    ]
