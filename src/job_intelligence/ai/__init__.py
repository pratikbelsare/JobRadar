"""Provider-independent AI contracts and the Bedrock implementation."""

from .analysis import to_job_analysis
from .bedrock import BedrockProvider
from .errors import (
    AIConfigurationError,
    AIProviderError,
    AIResponseError,
    AIServiceError,
    AIThrottledError,
    AITimeoutError,
    AIValidationError,
)
from .local_embeddings import HuggingFaceEmbeddingProvider
from .mantle import BedrockMantleProvider, MantleClient, SignedMantleClient
from .models import (
    AIResponseMetadata,
    AIResult,
    CandidateProfileProposal,
    EmbeddingVector,
    ExplanationContext,
    ExplanationReference,
    ExplanationSource,
    JobMatchExplanation,
    JobRequirements,
)
from .profile import merge_candidate_profile_proposal
from .protocols import AIProvider, TextEmbeddingProvider
from .providers import (
    AIProviders,
    build_ai_providers,
    build_embedding_provider,
    build_llm_provider,
)

__all__ = [
    "AIConfigurationError",
    "AIProvider",
    "AIProviderError",
    "AIResponseError",
    "AIResponseMetadata",
    "AIResult",
    "AIServiceError",
    "AIThrottledError",
    "AITimeoutError",
    "AIValidationError",
    "BedrockProvider",
    "BedrockMantleProvider",
    "HuggingFaceEmbeddingProvider",
    "MantleClient",
    "SignedMantleClient",
    "AIProviders",
    "build_ai_providers",
    "build_embedding_provider",
    "build_llm_provider",
    "CandidateProfileProposal",
    "EmbeddingVector",
    "ExplanationContext",
    "ExplanationReference",
    "ExplanationSource",
    "JobRequirements",
    "JobMatchExplanation",
    "TextEmbeddingProvider",
    "merge_candidate_profile_proposal",
    "to_job_analysis",
]
