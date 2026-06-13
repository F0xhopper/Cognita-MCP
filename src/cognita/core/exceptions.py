class CognitaError(Exception):
    """Base for all domain errors."""


class NotFoundError(CognitaError):
    def __init__(self, resource: str, identifier: str | int) -> None:
        super().__init__(f"{resource} not found: {identifier}")
        self.resource = resource
        self.identifier = identifier


class ConflictError(CognitaError):
    pass


class IngestionError(CognitaError):
    pass


class StorageError(CognitaError):
    pass


class EmbeddingError(CognitaError):
    pass


class UnsupportedFormatError(CognitaError):
    def __init__(self, fmt: str) -> None:
        super().__init__(f"Unsupported file format: {fmt}")
        self.fmt = fmt


class UrlFetchError(CognitaError):
    pass
