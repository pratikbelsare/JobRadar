"""Safe application of resume-derived proposals to editable profiles."""

from __future__ import annotations

from ..models import CandidateProfile
from .models import CandidateProfileProposal


def merge_candidate_profile_proposal(
    profile: CandidateProfile,
    proposal: CandidateProfileProposal,
) -> CandidateProfile:
    """Fill empty profile fields without replacing configured manual values."""

    values = profile.model_dump(mode="python")
    proposal_values = proposal.model_dump(exclude_none=True, mode="python")
    for field, proposed_value in proposal_values.items():
        current_value = getattr(profile, field)
        if _is_empty_profile_value(current_value):
            values[field] = proposed_value
    return CandidateProfile.model_validate(values)


def _is_empty_profile_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, str)):
        return not value
    return False


__all__ = ["merge_candidate_profile_proposal"]
