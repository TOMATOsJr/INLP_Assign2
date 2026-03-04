# 2.1 + 2.2 (task 2):
# - 2.1: Analogy predictions for SVD, Word2Vec, and GloVe
# - 2.2 task 2: Pairwise cosine bias scores for GloVe only

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


ANALOGY_CASES: List[Tuple[str, str, str, str]] = [
    ("1. paris : france :: delhi : ? (Syntactic/Capital)", "paris", "france", "delhi"),
    ("2. king : man :: queen : ? (Semantic/Gender)", "king", "man", "queen"),
    ("3. swim : swimming :: run : ? (Syntactic/Tense)", "swim", "swimming", "run"),
]

BIAS_TARGETS: List[str] = ["doctor", "nurse", "homemaker"]
MALE_ANCHOR = "man"
FEMALE_ANCHOR = "woman"

DEFAULT_SVD_PATH = "embeddings/svd.pt"
DEFAULT_WORD2VEC_PATH = "embeddings/word2vec2.pt"
DEFAULT_GLOVE_NAME = "glove-wiki-gigaword-100"
DEFAULT_CANDIDATES = 50
DEFAULT_TOPN = 5


class MatrixAdapter:
    def __init__(self, name: str, vocab_index: Dict[str, int], index_vocab: Dict[int, str], matrix: np.ndarray):
        self.name = name
        self.vocab_index = vocab_index
        self.index_vocab = {int(idx): token for idx, token in index_vocab.items()}
        matrix = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self.matrix = matrix / norms

    def has_token(self, token: str) -> bool:
        return token in self.vocab_index

    def get_vector(self, token: str) -> np.ndarray:
        return self.matrix[self.vocab_index[token]]

    def most_similar_by_vector(self, vector: np.ndarray, topn: int) -> Tuple[List[str], List[float]]:
        if topn <= 0:
            return [], []
        query = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return [], []
        query = query / norm

        similarities = self.matrix @ query
        topn = min(topn, similarities.shape[0])
        if topn <= 0:
            return [], []

        top_indices = np.argpartition(similarities, -topn)[-topn:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        tokens = [self.index_vocab[int(idx)] for idx in top_indices]
        scores = [float(similarities[int(idx)]) for idx in top_indices]
        return tokens, scores


class GloVeAdapter:
    def __init__(self, keyed_vectors, name: str):
        self.name = name
        self.kv = keyed_vectors

    def has_token(self, token: str) -> bool:
        return token in self.kv.key_to_index

    def get_vector(self, token: str) -> np.ndarray:
        return np.asarray(self.kv.get_vector(token), dtype=np.float32)

    def most_similar_by_vector(self, vector: np.ndarray, topn: int) -> Tuple[List[str], List[float]]:
        if topn <= 0:
            return [], []
        result = self.kv.similar_by_vector(np.asarray(vector, dtype=np.float32), topn=topn)
        tokens = [token for token, _ in result]
        scores = [float(score) for _, score in result]
        return tokens, scores


def analogy_topk(adapter, a: str, b: str, c: str, candidates: int = 50, topn: int = 5):
    missing = [word for word in (a, b, c) if not adapter.has_token(word)]
    if missing:
        return missing, []

    vector = adapter.get_vector(b) - adapter.get_vector(a) + adapter.get_vector(c)
    vector = vector / (np.linalg.norm(vector) + 1e-12)

    candidate_tokens, candidate_scores = adapter.most_similar_by_vector(vector, topn=candidates)
    blocked = {a, b, c}
    filtered = [(token, score) for token, score in zip(candidate_tokens, candidate_scores) if token not in blocked]
    return [], filtered[:topn]


def evaluate_model(adapter, analogy_cases: Sequence[Tuple[str, str, str, str]], candidates: int, topn: int):
    print(f"\n==== {adapter.name} ====")
    print("Top 5 analogy predictions:")
    for label, a, b, c in analogy_cases:
        missing_words, results = analogy_topk(adapter, a, b, c, candidates=candidates, topn=topn)
        print(label)
        if missing_words:
            print("Missing words in vocabulary:", ", ".join(missing_words))
        else:
            for token, score in results:
                print(f"  {token}\t{score:.4f}")
        print("---")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    vec_a = np.asarray(vec_a, dtype=np.float32)
    vec_b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def evaluate_glove_bias_pairwise(glove_adapter):
    print(f"\n==== {glove_adapter.name} ====")
    print("2.2 Task 2 - Pairwise cosine similarity bias check")

    required = [MALE_ANCHOR, FEMALE_ANCHOR] + BIAS_TARGETS
    missing = [word for word in required if not glove_adapter.has_token(word)]
    if missing:
        print("Missing words in vocabulary:", ", ".join(missing))
        print("---")
        return

    man_vec = glove_adapter.get_vector(MALE_ANCHOR)
    woman_vec = glove_adapter.get_vector(FEMALE_ANCHOR)

    for target in BIAS_TARGETS:
        target_vec = glove_adapter.get_vector(target)
        score_man = cosine_similarity(target_vec, man_vec)
        score_woman = cosine_similarity(target_vec, woman_vec)
        print(f"{target}: cos({target}, {MALE_ANCHOR}) = {score_man:.4f}, cos({target}, {FEMALE_ANCHOR}) = {score_woman:.4f}")
    print("---")


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_svd_adapter(path: Path) -> MatrixAdapter:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required to load saved .pt checkpoints. Install with: pip install torch") from exc

    data = torch.load(str(path), weights_only=False)
    return MatrixAdapter(
        name=f"SVD ({path.name})",
        vocab_index=data["vocab_index"],
        index_vocab=data["index_vocab"],
        matrix=data["embeddings"],
    )


def load_word2vec_adapter(path: Path) -> MatrixAdapter:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required to load saved .pt checkpoints. Install with: pip install torch") from exc

    data = torch.load(str(path), weights_only=False)
    return MatrixAdapter(
        name=f"Word2Vec ({path.name})",
        vocab_index=data["vocab_index"],
        index_vocab=data["index_vocab"],
        matrix=data["embeddings"],
    )


def load_glove_adapter(glove_name: str) -> GloVeAdapter:
    try:
        import importlib
        api = importlib.import_module("gensim.downloader")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gensim is required for pretrained GloVe. Install with: pip install gensim"
        ) from exc

    print(f"Loading pretrained embeddings: {glove_name}")
    vectors = api.load(glove_name)
    return GloVeAdapter(vectors, name=f"GloVe ({glove_name})")


def main():
    script_dir = Path(__file__).resolve().parent

    svd_path = _resolve_path(script_dir, DEFAULT_SVD_PATH)
    word2vec_path = _resolve_path(script_dir, DEFAULT_WORD2VEC_PATH)

    svd_adapter = load_svd_adapter(svd_path)
    word2vec_adapter = load_word2vec_adapter(word2vec_path)
    glove_adapter = load_glove_adapter(DEFAULT_GLOVE_NAME)

    print("2.1 - Analogy evaluation")
    evaluate_model(svd_adapter, ANALOGY_CASES, candidates=DEFAULT_CANDIDATES, topn=DEFAULT_TOPN)
    evaluate_model(word2vec_adapter, ANALOGY_CASES, candidates=DEFAULT_CANDIDATES, topn=DEFAULT_TOPN)
    evaluate_model(glove_adapter, ANALOGY_CASES, candidates=DEFAULT_CANDIDATES, topn=DEFAULT_TOPN)

    print("\n2.2 - Task 2 Bias check")
    evaluate_glove_bias_pairwise(glove_adapter)


if __name__ == "__main__":
    main()
