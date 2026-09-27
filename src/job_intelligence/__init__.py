"""Core package for the Job Intelligence platform."""

__version__ = "0.1.0"

from .config import Settings
from .change_detection import (
    JobChangeDetector,
    JobChangeResult,
    JobChangeType,
    StoredJobState,
)
from .deduplication import ExactDuplicateDetector
from .ingestion import JobIngestionService
from .job_repository import JobRepository, JsonJobRepository, JobRepositoryError
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
from .normalization import (
    JobIdentity,
    JobNormalizer,
    canonicalize_url,
    compute_content_hash,
    job_identity,
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
from .snapshots import (
    FileRawSnapshotStore,
    RawSnapshotMetadata,
    RawSnapshotStore,
    SnapshotNotFoundError,
    SnapshotStoreError,
)

__all__ = [
    "CandidateProfile",
    "CandidateProfileRepository",
    "CandidateProfileService",
    "Company",
    "ExactDuplicateDetector",
    "FileRawSnapshotStore",
    "JobChangeDetector",
    "JobChangeResult",
    "JobChangeType",
    "JobIdentity",
    "JobIngestionService",
    "JobNormalizer",
    "JobRepository",
    "JobRepositoryError",
    "Job",
    "JobAnalysis",
    "JobMatch",
    "JobVersion",
    "JsonJobRepository",
    "JsonCandidateProfileRepository",
    "PdfTextExtractor",
    "ProfileAlreadyExistsError",
    "ProfileNotFoundError",
    "ProfileRepositoryError",
    "RawSnapshotMetadata",
    "RawSnapshotStore",
    "ResumeDocument",
    "ResumeIngestionError",
    "ResumeIngestor",
    "ResumeMetadata",
    "ScanRun",
    "Settings",
    "SnapshotNotFoundError",
    "SnapshotStoreError",
    "StoredJobState",
    "UserFeedback",
    "canonicalize_url",
    "compute_content_hash",
    "job_identity",
]
