"""Build an explicit, score-independent Benchmark v2 source plan."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from job_intelligence.benchmark import write_json
from job_intelligence.benchmark_v2 import (
    BenchmarkSource,
    BenchmarkSourcePlan,
    SelectionBucket,
    build_source_plan_jobs,
    selection_bucket,
)
from job_intelligence.connectors import ConnectorStatus, GreenhouseConnector
from job_intelligence.http import HttpClient
from job_intelligence.models import Company, ConnectorType

SOURCE_SPECS = (
    ("InMobi", "inmobi"),
    ("Moloco", "moloco"),
    ("Dialpad", "dialpad"),
    ("Samaya AI", "samayaai"),
    ("Garner Health", "garnerhealth"),
    ("PhysicsX", "physicsx"),
    ("Together AI", "togetherai"),
    ("Snorkel AI", "snorkelai"),
)
TARGET_COMPOSITION = {
    SelectionBucket.TARGET_REGULAR: 17,
    SelectionBucket.TARGET_STRETCH: 18,
    SelectionBucket.ADJACENT: 20,
    SelectionBucket.IRRELEVANT: 29,
}


def main() -> None:
    args = _arguments()
    client = HttpClient(
        timeout_seconds=args.timeout,
        user_agent=args.user_agent,
        max_retries=1,
    )
    connector = GreenhouseConnector(client)
    sources = [_source(name, token) for name, token in SOURCE_SPECS]
    listings = {}
    for source in sources:
        result = connector.list_jobs(_company(source))
        if result.status != ConnectorStatus.SUCCESS:
            raise RuntimeError(
                f"could not list {source.company_name}: {result.status.value} {result.error}"
            )
        listings[source.company_id] = result.jobs
        unique_titles = {job.title.casefold(): job for job in result.jobs}.values()
        inventory = Counter(selection_bucket(job.title).value for job in unique_titles)
        print(
            f"listed company={source.company_name} jobs={len(result.jobs)} "
            f"buckets={dict(sorted(inventory.items()))}",
            flush=True,
        )
    jobs = build_source_plan_jobs(
        sources,
        listings,
        target_composition=TARGET_COMPOSITION,
        max_jobs_per_company=args.max_jobs_per_company,
    )
    plan = BenchmarkSourcePlan(
        created_at=datetime.now(timezone.utc),
        target_size=sum(TARGET_COMPOSITION.values()),
        max_jobs_per_company=args.max_jobs_per_company,
        target_composition=TARGET_COMPOSITION,
        sources=sources,
        jobs=jobs,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, plan.model_dump(mode="json"))
    _write_review(output.with_name("selection_review.csv"), plan)
    print(json.dumps(_summary(plan), sort_keys=True))


def _source(name: str, token: str) -> BenchmarkSource:
    career_url = f"https://job-boards.greenhouse.io/{token}"
    return BenchmarkSource.model_validate(
        {
            "company_name": name,
            "board_token": token,
            "career_url": career_url,
            "company_id": uuid5(NAMESPACE_URL, f"jobradar:greenhouse:{token}"),
        }
    )


def _company(source: BenchmarkSource) -> Company:
    return Company(
        id=source.company_id,
        name=source.company_name,
        career_url=source.career_url,
        connector_type=ConnectorType.GREENHOUSE,
        connector_config={"board_token": source.board_token},
    )


def _write_review(path: Path, plan: BenchmarkSourcePlan) -> None:
    fields = (
        "company_name",
        "source_job_id",
        "title",
        "location",
        "role_category",
        "selection_bucket",
        "canonical_url",
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for job in plan.jobs:
            writer.writerow(
                {
                    **job.model_dump(mode="json", include=set(fields)),
                    "selection_bucket": job.selection_bucket.value,
                }
            )


def _summary(plan: BenchmarkSourcePlan) -> dict[str, object]:
    companies = {
        source.company_name: sum(
            job.company_id == source.company_id for job in plan.jobs
        )
        for source in plan.sources
    }
    buckets = {
        bucket.value: sum(job.selection_bucket == bucket for job in plan.jobs)
        for bucket in SelectionBucket
    }
    return {"jobs": len(plan.jobs), "companies": companies, "buckets": buckets}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="evaluation/benchmark/v2/source_plan.json")
    parser.add_argument("--max-jobs-per-company", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--user-agent", default="JobRadar-Benchmark/2.0")
    return parser.parse_args()


if __name__ == "__main__":
    main()
