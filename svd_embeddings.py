from embeddings.base import EmbeddingModel
from nltk.corpus import brown
import numpy as np
from scipy.sparse import dok_matrix, coo_matrix, csr_matrix


class SVDEmbedding(EmbeddingModel):
	"""Boilerplate SVD-backed embedding (surface only)."""

	def __init__(
			self,
			n_components: int = 100,
			window_size: int = 2,
	):
		self.n_components = n_components # number of dimensions for the embedding
		self.window_size = window_size # context window size


	def get_cooccurrence_matrix(self, corpus):
		"""Compute co-occurrence matrix from corpus."""
		from collections import Counter

		token_counts = Counter(token for sentence in corpus for token in sentence)
		vocab = list(token_counts.keys())
		vocab_index = {token: idx for idx, token in enumerate(vocab)}
		index_vocab = {idx: token for token, idx in vocab_index.items()}
		vocab_size = len(vocab)
		cooc_matrix = dok_matrix((vocab_size, vocab_size), dtype=np.float32)

		# Now fill up the co-occurrence matrix
		from tqdm import tqdm
		for sentence in tqdm(corpus, desc="Building co-occurrence matrix"):
			tokens = [t for t in sentence if t in vocab_index]
			for i, token in enumerate(tokens):
				token_idx = vocab_index[token]
				# Look at the context window around the token
				# Window start and end indices
				start = max(0, i - self.window_size)
				end = min(len(tokens), i + self.window_size + 1)
				for j in range(start, end):
					if j != i:
						context_token = tokens[j]
						context_token_idx = vocab_index[context_token]
						# Increment the co-occurrence count for this token-context pair
						# TODO:Later try dividing by distance or applying weighting schemes
						distance = abs(i - j)
						weight = 1.0 / distance # simple inverse distance weighting
						cooc_matrix[token_idx, context_token_idx] += weight

		self.vocab_index = vocab_index
		self.index_vocab = index_vocab
		return cooc_matrix.tocsr(), vocab_index

	def _compute_ppmi(self, cooc_matrix):
		"""
		Compute PPMI matrix from co-occurrence matrix.
		TODO: Check this implementation.
		"""
		if not isinstance(cooc_matrix, csr_matrix):
			cooc_matrix = cooc_matrix.tocsr()

		total = float(cooc_matrix.sum())
		if total <= 0.0:
			return cooc_matrix

		row_sum = np.asarray(cooc_matrix.sum(axis=1)).ravel()
		col_sum = np.asarray(cooc_matrix.sum(axis=0)).ravel()

		coo = cooc_matrix.tocoo(copy=True)
		eps = 1e-10
		denom = (row_sum[coo.row] * col_sum[coo.col]) + eps
		pmi = np.log((coo.data * total) / denom + eps)
		ppmi_data = np.maximum(pmi, 0.0).astype(np.float32)

		ppmi = coo_matrix((ppmi_data, (coo.row, coo.col)), shape=cooc_matrix.shape, dtype=np.float32).tocsr()
		ppmi.eliminate_zeros()
		return ppmi

	def train(self, corpus, **kwargs):
		"""Train from corpus (not implemented)."""
		from sklearn.decomposition import TruncatedSVD

		cooc_matrix, vocab_index = self.get_cooccurrence_matrix(corpus)
		cooc_matrix = self._compute_ppmi(cooc_matrix)
		cooc_matrix = csr_matrix(cooc_matrix)

		# Perform SVD on the co-occurrence matrix
		svd = TruncatedSVD(
			n_components=self.n_components,
			random_state=42
		)

		embeddings = svd.fit_transform(cooc_matrix)
		# Normalize embeddings for cosine similarity
		norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
		norms = np.where(norms == 0.0, 1.0, norms)
		embeddings = embeddings / norms

		# if np.any(np.isnan(embeddings)):
		# 	print("Warning: NaN values found in embeddings after SVD. Check co-occurrence matrix and PPMI computation.")
		# if np.any(np.isinf(embeddings)):
		# 	print("Warning: Inf values found in embeddings after SVD. Check co-occurrence matrix and PPMI computation.")

		# map embeddings to vocab
		self.embeddings = embeddings

	def get_vector(self, token):
		"""Return embedding vector for a token."""
		if token not in self.vocab_index:
			raise ValueError(f"Token '{token}' not in vocabulary.")
		idx = self.vocab_index[token]
		return self.embeddings[idx]

	def most_similar(self, query, topn=1):
		"""Return most similar tokens to the query token."""
		if query not in self.vocab_index:
			raise ValueError(f"Token '{query}' not in vocabulary.")
		query_idx = self.vocab_index[query]
		query_vector = self.embeddings[query_idx]
		# Compute cosine similarity with all other vectors
		similarities = self.embeddings @ query_vector # dot product since vectors are normalized

		# Exclude the query token itself
		similarities[query_idx] = -np.inf

		# Get topn most similar tokens
		top_indices = np.argsort(similarities)[-topn:][::-1]

		similar_tokens = []
		scores = []
		for idx in top_indices:
			token = self.index_vocab[idx]
			similarity = similarities[idx]
			similar_tokens.append(token)
			scores.append(similarity)

		return similar_tokens, scores

	def save(self, path):
		"""Save the model as a .pt file."""
		import torch
		try:
			torch.save({
				'embeddings': self.embeddings,
				'vocab_index': self.vocab_index,
				'index_vocab': self.index_vocab,
				'n_components': self.n_components,
				'window_size': self.window_size
			}, path)
			print(f"Model saved to {path}")
		except Exception as e:
			print(f"Error saving model: {e} to {path}")


	@classmethod
	def load(cls, path):
		"""Load the model from a .pt file."""
		import torch
		try:
			data = torch.load(path, weights_only=False)
			model = cls(
				n_components=data['n_components'],
				window_size=data['window_size']
			)
			model.embeddings = data['embeddings']
			model.vocab_index = data['vocab_index']
			model.index_vocab = data['index_vocab']
			print(f"Model loaded from {path}")
		except Exception as e:
			print(f"Error loading model: {e} from {path}")
			raise e
		return model

	def vocab(self):
		"""Return an iterable of vocabulary tokens."""
		return self.vocab_index.keys()

	def dimension(self):
		return self.n_components

	def nearest_neighbors(self, vector, topn=10):
		raise NotImplementedError

if __name__ == "__main__":
	# Example usage
	corpus = [[token.lower() for token in sentence] for sentence in brown.sents()]
	# model = SVDEmbedding(n_components=100, window_size=2)
	# model.train(corpus)
	# model.save("./embeddings/svd.pt")
	# model.save("./Assign2/embeddings/svd.pt")
	model = SVDEmbedding.load("./embeddings/svd.pt")

	tokens, scores = model.most_similar("run", topn=10)

	for token, score in zip(tokens, scores):
		print(f"Token: {token}, Similarity: {score}")

	# # print some embeddings for a few words
	# words_to_check = ["the", "man", "woman", "dog", "city"]
	# for word in words_to_check:
	# 	if word in model.embeddings:
	# 		print(f"Embedding for '{word}': {model.embeddings[word]}")
	# 	else:
	# 		print(f"'{word}' not in vocabulary.")

	# print("Co-occurrence matrix shape:", cooc_matrix.shape)
	# print("Co-occurrence matrix:\n", cooc_matrix)
	# print("Vocabulary size:", len(vocab_index))