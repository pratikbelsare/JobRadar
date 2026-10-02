"""Core package for the Job Intelligence platform."""

__version__ = "0.1.0"

from .change_detection import (
    JobChangeDetector,
    JobChangeResult,
    JobChangeType,
    StoredJobState,
)
from .config import Settings
from .deduplication import ExactDuplicateDetector
from .evaluation import (
    EmbeddingBaseline,
    EvaluationCase,
    EvaluationExample,
    EvaluationReport,
    HybridRankingBaseline,
    JsonlEvaluationDataset,
    KeywordBaseline,
    evaluate_approaches,
    ndcg_at_k,
    precision_at_k,
)
from .explanations import (
    ExplanationGenerationError,
    ExplanationGroundingError,
    ExplanationService,
)
from .feedback import (
    FeedbackAlreadyExistsError,
    FeedbackNotFoundError,
    FeedbackRepositoryError,
    JsonUserFeedbackRepository,
    UserFeedbackService,
)
from .filtering import (
    DeterministicJobFilter,
    ExperienceFilterConfig,
    FilteringConfig,
    FilterOutcome,
    FilterResult,
    FilterRule,
    FilterRuleResult,
    RuleStatus,
)
from .ingestion import JobIngestionService
from .job_repository import JobRepository, JobRepositoryError, JsonJobRepository
from .matching import (
    HybridJobMatcher,
    MatchComponent,
    MatchingConfig,
    MatchingWeights,
    RejectedJobError,
)
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
    "EmbeddingBaseline",
    "DeterministicJobFilter",
    "ExperienceFilterConfig",
    "EvaluationCase",
    "EvaluationExample",
    "EvaluationReport",
    "ExplanationGenerationError",
    "ExplanationGroundingError",
    "ExplanationService",
    "FilterOutcome",
    "FilterResult",
    "FilterRule",
    "FilterRuleResult",
    "FilteringConfig",
    "FileRawSnapshotStore",
    "FeedbackAlreadyExistsError",
    "FeedbackNotFoundError",
    "FeedbackRepositoryError",
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
    "HybridJobMatcher",
    "HybridRankingBaseline",
    "JsonlEvaluationDataset",
    "JsonUserFeedbackRepository",
    "KeywordBaseline",
    "MatchComponent",
    "MatchingConfig",
    "MatchingWeights",
    "JsonJobRepository",
    "JsonCandidateProfileRepository",
    "PdfTextExtractor",
    "ProfileAlreadyExistsError",
    "ProfileNotFoundError",
    "ProfileRepositoryError",
    "RawSnapshotMetadata",
    "RawSnapshotStore",
    "RejectedJobError",
    "ResumeDocument",
    "ResumeIngestionError",
    "ResumeIngestor",
    "ResumeMetadata",
    "RuleStatus",
    "ScanRun",
    "Settings",
    "SnapshotNotFoundError",
    "SnapshotStoreError",
    "StoredJobState",
    "UserFeedback",
    "UserFeedbackService",
    "evaluate_approaches",
    "ndcg_at_k",
    "precision_at_k",
    "canonicalize_url",
    "compute_content_hash",
    "job_identity",
]
