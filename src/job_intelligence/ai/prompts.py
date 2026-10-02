"""Small versioned prompts kept separate from provider and business code."""

from __future__ import annotations

import json

from .models import ExplanationContext

JOB_REQUIREMENTS_PROMPT_VERSION = "job-requirements-v1"
CANDIDATE_PROFILE_PROMPT_VERSION = "candidate-profile-v1"
EXPLANATION_PROMPT_VERSION = "job-match-explanation-v1"

JOB_REQUIREMENTS_SYSTEM_PROMPT = """Extract only facts explicitly supported by the job description.
Return one JSON object with the requested schema. Use null or empty arrays when a fact is
not stated. Do not infer candidate suitability, and do not invent skills, experience,
location constraints, or education requirements. Use exactly these enum values: seniority
is intern, entry, associate, mid, senior, lead, principal, manager, director, or unknown;
work_modes are remote, hybrid, onsite, flexible, or unknown; employment_type is full_time,
part_time, contract, internship, temporary, or unknown. Use unknown rather than null for
enum fields, and use one string or null for domain. The fields required_skills,
preferred_skills, responsibilities, education, location_constraints, and work_modes must
always be JSON arrays, using an empty array when unsupported. The experience fields must
be numbers or null."""

CANDIDATE_PROFILE_SYSTEM_PROMPT = """Extract only facts explicitly supported by the resume.
Return one JSON object with the requested schema. Use null for unknown scalar values and
empty or null collections when unsupported. Preserve evidence text where available. Do
not claim that a skill is absent merely because no evidence appears."""

EXPLANATION_SYSTEM_PROMPT = """Generate a structured explanation using only the supplied job,
candidate, and match evidence. Do not recalculate or change any score. Do not invent
experience or requirements. If evidence for a skill is absent, say 'no evidence found'
instead of claiming the candidate lacks the skill. Include evidence references for
positive claims where possible. Each evidence_references item must be an object with
source (job, candidate, or match), claim, evidence, and optional string reference_id fields;
do not return evidence references as strings. why_fit, strongest_matches,
potential_gaps, and important_considerations must be JSON arrays of strings;
experience_alignment must be one string. The evidence value in each reference must be
copied exactly from the supplied evidence context, not paraphrased; omit a reference
when an exact evidence substring is unavailable. Never say that the candidate lacks a
skill, does not know a skill, or has no skill. For an unsupported gap, use the exact
phrase 'no evidence found' instead. Keep each list to at most three concise items and
return JSON only."""


def job_requirements_prompt(job_text: str) -> str:
    return _json_prompt(
        "job requirements",
        job_text,
        "min_experience_years, max_experience_years, seniority, required_skills, "
        "preferred_skills, responsibilities, education, location_constraints, "
        "work_modes, employment_type, domain",
    )


def candidate_profile_prompt(resume_text: str) -> str:
    return _json_prompt(
        "candidate profile proposal",
        resume_text,
        "target_roles, experience_years, preferred_locations, work_mode, skills, "
        "skill_evidence, work_experience, projects, education, domains",
    )


def explanation_prompt(context: ExplanationContext) -> str:
    return (
        "Create a grounded job-match explanation with fields: why_fit, "
        "strongest_matches, potential_gaps, experience_alignment, "
        "important_considerations, evidence_references.\n"
        "Return JSON only. Here is the complete evidence context:\n"
        f"{json.dumps(context.model_dump(mode='json'), sort_keys=True)}"
    )


def _json_prompt(subject: str, text: str, fields: str) -> str:
    return (
        f"Extract a structured {subject}. Fields: {fields}.\n"
        "Return JSON only.\n\nSource text:\n"
        f"{text.strip()}"
    )


__all__ = [
    "CANDIDATE_PROFILE_PROMPT_VERSION",
    "JOB_REQUIREMENTS_PROMPT_VERSION",
    "CANDIDATE_PROFILE_SYSTEM_PROMPT",
    "EXPLANATION_PROMPT_VERSION",
    "EXPLANATION_SYSTEM_PROMPT",
    "JOB_REQUIREMENTS_SYSTEM_PROMPT",
    "candidate_profile_prompt",
    "explanation_prompt",
    "job_requirements_prompt",
]
