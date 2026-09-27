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

__all__ = [
    "CandidateProfile",
    "Company",
    "Job",
    "JobAnalysis",
    "JobMatch",
    "JobVersion",
    "ScanRun",
    "Settings",
    "UserFeedback",
]
