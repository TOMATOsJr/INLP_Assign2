"""Nearest-neighbor utilities (boilerplate).

This module contains only a placeholder API surface. Implementations
should provide efficient nearest-neighbor search (exact or approximate).
"""

from typing import List, Tuple


def nearest_neighbors(vectors, query_vector, topn: int = 10, method: str = "exact") -> List[Tuple[int, float]]:
    """Return list of (index, score) tuples for the nearest neighbors.

    Not implemented — boilerplate only.
    """
    raise NotImplementedError
