"""Public facetorch exception hierarchy."""


class FacetorchError(Exception):
    """Base class for actionable facetorch errors."""


class InputError(FacetorchError, ValueError):
    """The caller supplied an unsupported or ambiguous image input."""


class ConfigurationError(FacetorchError, ValueError):
    """The configured component graph or option set is invalid."""


class ModelCompatibilityError(FacetorchError, RuntimeError):
    """No model artifact is compatible with the active runtime."""


class OfflineCacheError(FacetorchError, FileNotFoundError):
    """Offline execution was requested but a required cache entry is absent."""


class CacheLockError(FacetorchError, TimeoutError):
    """A model-cache operation could not acquire its process lock."""


class ArtifactIntegrityError(FacetorchError, RuntimeError):
    """A model or distribution artifact failed integrity verification."""


class InferenceError(FacetorchError, RuntimeError):
    """A configured model failed while executing inference."""


class InputCoercionWarning(UserWarning):
    """A deterministic input conversion was performed in ``coerce`` mode."""


class LegacyModelWarning(UserWarning):
    """An explicitly enabled legacy TorchScript artifact was selected."""
