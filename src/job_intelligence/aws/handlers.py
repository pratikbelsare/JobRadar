"""Thin AWS Lambda entry points that delegate to testable workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from .diagnostics import log_processing_failure
from .runtime import build_analysis_worker, build_coordinator, build_scanner
from .sqs import parse_company_scan_message, parse_job_analysis_message
from .workers import CompanyScanner, JobAnalysisWorker, RunCoordinator


def coordinator_handler(
    event: Mapping[str, Any],
    context: Any,
    *,
    coordinator: RunCoordinator | None = None,
) -> dict[str, Any]:
    del event, context
    result = (coordinator or build_coordinator()).start()
    return {
        "run_id": str(result.run.id),
        "company_ids": [str(company_id) for company_id in result.selected_company_ids],
    }


def scanner_handler(
    event: Mapping[str, Any],
    context: Any,
    *,
    scanner: CompanyScanner | None = None,
) -> dict[str, list[dict[str, str]]]:
    del context
    worker = scanner or build_scanner()
    return _process_records(
        event,
        lambda body: worker.process(parse_company_scan_message(body)),
        context_for_body=_company_message_context,
    )


def analysis_handler(
    event: Mapping[str, Any],
    context: Any,
    *,
    worker: JobAnalysisWorker | None = None,
) -> dict[str, list[dict[str, str]]]:
    del context
    analysis_worker = worker or build_analysis_worker()
    return _process_records(
        event,
        lambda body: analysis_worker.process(parse_job_analysis_message(body)),
        context_for_body=_analysis_message_context,
    )


def _process_records(
    event: Mapping[str, Any],
    process: Callable[[str], Any],
    *,
    context_for_body: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, list[dict[str, str]]]:
    failures: list[dict[str, str]] = []
    records = event.get("Records", [])
    if not isinstance(records, list):
        raise ValueError("SQS Lambda event must contain a Records list")
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("SQS Lambda record must be an object")
        message_id = record.get("messageId")
        body = record.get("body")
        if not isinstance(message_id, str) or not isinstance(body, str):
            raise ValueError("SQS Lambda record must contain messageId and body")
        try:
            process(body)
        except Exception as exc:
            identifiers = context_for_body(body) if context_for_body is not None else {}
            log_processing_failure(
                event="sqs_record_failed",
                message_id=message_id,
                stage=str(identifiers.get("stage", "handler")),
                error=exc,
                identifiers=identifiers,
            )
            failures.append({"itemIdentifier": message_id})
    return {"batchItemFailures": failures}


def _company_message_context(body: str) -> Mapping[str, Any]:
    try:
        message = parse_company_scan_message(body)
    except Exception:
        return {"stage": "message_parse"}
    return {
        "stage": "worker",
        "run_id": message.run_id,
        "company_id": message.company_id,
    }


def _analysis_message_context(body: str) -> Mapping[str, Any]:
    try:
        message = parse_job_analysis_message(body)
    except Exception:
        return {"stage": "message_parse"}
    return {
        "stage": "worker",
        "run_id": message.run_id,
        "job_id": message.job_id,
        "candidate_profile_id": message.candidate_profile_id,
    }


__all__ = ["analysis_handler", "coordinator_handler", "scanner_handler"]
