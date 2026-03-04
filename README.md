## SVD Embeddings Notes

- `SVDEmbedding` now supports `min_freq=0`, which keeps all observed tokens (no frequency filtering).
- Co-occurrence and PPMI are computed with sparse matrices (`scipy.sparse`) to reduce memory usage during training.
- `TruncatedSVD` still outputs dense final embeddings, so final model size depends on `vocab_size x n_components`.

## Evaluation Script (Section 2.1 + 2.2 Task 2)

- Script: `analogy_eval.py`
- Default checkpoints:
	- SVD: `embeddings/svd.pt`
	- Word2Vec: `embeddings/word2vec2.pt`
- Pretrained embeddings: GloVe via `gensim.downloader` (`glove-wiki-gigaword-100`)
- What it runs:
	- 2.1 analogy predictions on SVD + Word2Vec + GloVe
	- 2.2 task 2 pairwise cosine bias scores on GloVe only:
		- `cos(doctor, man)` vs `cos(doctor, woman)`
		- `cos(nurse, man)` vs `cos(nurse, woman)`
		- `cos(homemaker, man)` vs `cos(homemaker, woman)`

Install dependency (if needed):

```bash
pip install gensim
```

Run with defaults:

```bash
python analogy_eval.py
```
