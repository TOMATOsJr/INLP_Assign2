## SVD Embeddings Notes

- `SVDEmbedding` now supports `min_freq=0`, which keeps all observed tokens (no frequency filtering).
- Co-occurrence and PPMI are computed with sparse matrices (`scipy.sparse`) to reduce memory usage during training.
- `TruncatedSVD` still outputs dense final embeddings, so final model size depends on `vocab_size x n_components`.
