"""Core package for the Job Intelligence platform."""

__version__ = "0.1.0"

from .config import Settings
from .models import (
    CandidateProfile,
    Company,
    Job,
    JobAnalysis,
    JobMatch,
    JobVersion,
    ScanRun,
    UserFeedback,
)
from .profile_repository import (
    CandidateProfileRepository,
    JsonCandidateProfileRepository,
    ProfileRepositoryError,
)
from .profile_service import (
    CandidateProfileService,
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
)
from .resume import (
    PdfTextExtractor,
    ResumeDocument,
    ResumeIngestionError,
    ResumeIngestor,
    ResumeMetadata,
)

__all__ = [
    "CandidateProfile",
    "CandidateProfileRepository",
    "CandidateProfileService",
    "Company",
    "Job",
    "JobAnalysis",
    "JobMatch",
    "JobVersion",
    "JsonCandidateProfileRepository",
    "PdfTextExtractor",
    "ProfileAlreadyExistsError",
    "ProfileNotFoundError",
    "ProfileRepositoryError",
    "ResumeDocument",
    "ResumeIngestionError",
    "ResumeIngestor",
    "ResumeMetadata",
    "ScanRun",
    "Settings",
    "UserFeedback",
]
