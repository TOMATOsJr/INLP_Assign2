from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Abstract embedding model interface (gensim-like surface).

    Methods are intentionally unimplemented — this file provides the
    minimal surface (boilerplate) only.
    """

    @abstractmethod
    def train(self, corpus, **kwargs):
        """Train the model from a corpus (or accept precomputed data)."""
        raise NotImplementedError

    @abstractmethod
    def get_vector(self, token):
        """Return the vector for a single token."""
        raise NotImplementedError

    @abstractmethod
    def most_similar(self, query, topn=10):
        """Return top-N most similar tokens to `query` (token or vector)."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        """Persist model to `path`."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Load and return a model instance from `path`."""
        raise NotImplementedError

    @abstractmethod
    def vocab(self):
        """Return an iterable of vocabulary tokens."""
        raise NotImplementedError

    @abstractmethod
    def dimension(self):
        """Return embedding dimensionality (int)."""
        raise NotImplementedError

    @abstractmethod
    def nearest_neighbors(self, vector, topn=10):
        """Return nearest neighbors for an arbitrary vector."""
        raise NotImplementedError
