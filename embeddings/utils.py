"""Nearest-neighbor utilities (boilerplate).

This module contains only a placeholder API surface. Implementations
should provide efficient nearest-neighbor search (exact or approximate).
"""

from typing import List, Tuple
import numpy as np


def nearest_neighbors(vectors, query_vector, topn: int = 10, method: str = "exact") -> List[Tuple[int, float]]:
    """Return list of (index, score) tuples for the nearest neighbors.

    Not implemented — boilerplate only.
    """
    raise NotImplementedError

def analogy_top5(a, b, c, model, candidates=50, topn=5):
		missing = [word for word in (a, b, c) if word not in model.vocab_index]
		if missing:
			return missing, []

		vector = model.get_vector(b) - model.get_vector(a) + model.get_vector(c)
		vector = vector / (np.linalg.norm(vector) + 1e-12)
		candidate_tokens, candidate_scores = model.most_similar(vector, topn=candidates)
		blocked = {a, b, c}
		filtered = [(token, score) for token, score in zip(candidate_tokens, candidate_scores) if token not in blocked]
		return [], filtered[:topn]