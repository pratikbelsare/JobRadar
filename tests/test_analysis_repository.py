from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from job_intelligence.analysis_repository import JsonJobAnalysisRepository
from job_intelligence.models import JobAnalysis


def test_json_analysis_repository_uses_exact_cache_key(tmp_path: Path) -> None:
    repository = JsonJobAnalysisRepository(tmp_path)
    analysis = JobAnalysis(
        job_id=uuid4(),
        job_version_id=uuid4(),
        model_id="model/a:1",
        prompt_version="prompt-v1",
    )

    repository.save(analysis)

    assert repository.get(
        analysis.job_id,
        analysis.job_version_id,
        analysis.model_id,
        analysis.prompt_version,
    ) == analysis
    assert repository.get(
        analysis.job_id,
        analysis.job_version_id,
        analysis.model_id,
        "prompt-v2",
    ) is None
