from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract interface for LLM generation backends."""

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate an answer for the given query using the provided context."""
        pass
