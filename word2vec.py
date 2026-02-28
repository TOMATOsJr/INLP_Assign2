from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import brown
from tqdm import tqdm

from embeddings.base import EmbeddingModel


class Word2VecEmbedding(EmbeddingModel):
	"""Skip-Gram with Negative Sampling (SGNS) implemented in PyTorch."""

	def __init__(
			self,
			embedding_dim: int = 100,
			window_size: int = 2,
			num_negatives: int = 5,
			epochs: int = 5,
			batch_size: int = 1024,
			learning_rate: float = 1e-2,
			vector_size: Optional[int] = None,
	):
		if vector_size is not None:
			embedding_dim = vector_size
		if embedding_dim <= 0:
			raise ValueError("embedding_dim must be > 0")
		if window_size <= 0:
			raise ValueError("window_size must be > 0")
		if num_negatives <= 0:
			raise ValueError("num_negatives must be > 0")
		if epochs <= 0:
			raise ValueError("epochs must be > 0")
		if batch_size <= 0:
			raise ValueError("batch_size must be > 0")
		if learning_rate <= 0:
			raise ValueError("learning_rate must be > 0")

		self.embedding_dim = int(embedding_dim)
		self.vector_size = int(embedding_dim)
		self.window_size = int(window_size)
		self.num_negatives = int(num_negatives)
		self.epochs = int(epochs)
		self.batch_size = int(batch_size)
		self.learning_rate = float(learning_rate)

		self.vocab_index = {}
		self.index_vocab = {}
		self.token_counts = None
		self.negative_sampling_probs = None
		self.embeddings = None
		self.input_embedding_table = None
		self.output_embedding_table = None

	def _build_vocab(self, corpus):
		token_counts = Counter(token for sentence in corpus for token in sentence)
		vocab_tokens = list(token_counts.keys())
		vocab_tokens = sorted(vocab_tokens, key=lambda token: (-token_counts[token], token))

		self.vocab_index = {token: idx for idx, token in enumerate(vocab_tokens)}
		self.index_vocab = {idx: token for token, idx in self.vocab_index.items()}
		self.token_counts = np.array([token_counts[token] for token in vocab_tokens], dtype=np.float64)

		if len(vocab_tokens) == 0:
			raise ValueError("No tokens in vocabulary")

		weights = np.power(self.token_counts, 0.75)
		weights_sum = float(weights.sum())
		if weights_sum <= 0.0:
			raise ValueError("Invalid negative sampling weights")
		self.negative_sampling_probs = torch.tensor(weights / weights_sum, dtype=torch.float32)

	def _index_corpus(self, corpus):
		indexed_corpus = []
		for sentence in corpus:
			indexed_sentence = [self.vocab_index[token] for token in sentence if token in self.vocab_index]
			if len(indexed_sentence) >= 2:
				indexed_corpus.append(indexed_sentence)
		return indexed_corpus

	def _iter_pairs(self, indexed_corpus):
		for sentence in indexed_corpus:
			for center_pos, center_token in enumerate(sentence):
				start = max(0, center_pos - self.window_size)
				end = min(len(sentence), center_pos + self.window_size + 1)
				for context_pos in range(start, end):
					if context_pos == center_pos:
						continue
					yield center_token, sentence[context_pos]

	def _count_pairs(self, indexed_corpus):
		total_pairs = 0
		for sentence in indexed_corpus:
			n = len(sentence)
			for center_pos in range(n):
				start = max(0, center_pos - self.window_size)
				end = min(n, center_pos + self.window_size + 1)
				total_pairs += (end - start - 1)
		return total_pairs

	def _train_batch(self, centers, contexts, input_embed, output_embed, optimizer, neg_probs, device):
		center_tensor = torch.tensor(centers, dtype=torch.long, device=device)
		context_tensor = torch.tensor(contexts, dtype=torch.long, device=device)
		batch_size = center_tensor.shape[0]

		neg_samples = torch.multinomial(
			neg_probs,
			num_samples=batch_size * self.num_negatives,
			replacement=True,
		).view(batch_size, self.num_negatives)

		v = input_embed(center_tensor)
		u_pos = output_embed(context_tensor)
		u_neg = output_embed(neg_samples)

		pos_logits = torch.sum(v * u_pos, dim=1)
		neg_logits = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)

		loss = -(F.logsigmoid(pos_logits).sum() + F.logsigmoid(-neg_logits).sum()) / batch_size

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		return float(loss.item())

	def train(self, corpus, **kwargs):
		if corpus is None:
			raise ValueError("corpus cannot be None")

		device_arg = kwargs.get("device", "cpu")
		device = torch.device(device_arg)
		optimizer_name = str(kwargs.get("optimizer", "adam")).lower()
		show_progress = bool(kwargs.get("show_progress", True))

		self._build_vocab(corpus)
		indexed_corpus = self._index_corpus(corpus)
		if len(indexed_corpus) == 0:
			raise ValueError("Indexed corpus is empty after vocabulary filtering")
		total_pairs = self._count_pairs(indexed_corpus)

		vocab_size = len(self.vocab_index)
		# Init embeddings
		input_embed = nn.Embedding(vocab_size, self.embedding_dim).to(device)
		output_embed = nn.Embedding(vocab_size, self.embedding_dim).to(device)
		# Fill them with small random values
		bound = 0.5 / self.embedding_dim
		nn.init.uniform_(input_embed.weight, -bound, bound)
		nn.init.zeros_(output_embed.weight)

		if optimizer_name == "sgd":
			optimizer = torch.optim.SGD(
				list(input_embed.parameters()) + list(output_embed.parameters()),
				lr=self.learning_rate,
			)
		else:
			optimizer = torch.optim.Adam(
				list(input_embed.parameters()) + list(output_embed.parameters()),
				lr=self.learning_rate,
			)

		if self.negative_sampling_probs is None:
			raise ValueError("Negative sampling probabilities are not initialized")
		neg_probs = self.negative_sampling_probs.to(device)

		for epoch in range(self.epochs):
			total_loss = 0.0
			batches = 0
			centers = []
			contexts = []
			pair_iter = self._iter_pairs(indexed_corpus)
			progress_bar = None
			if show_progress:
				desc = f"Epoch {epoch + 1}/{self.epochs}"
				progress_bar = tqdm(pair_iter, total=total_pairs, desc=desc, unit="pair")
				pair_iter = progress_bar

			for center, context in pair_iter:
				centers.append(center)
				contexts.append(context)
				if len(centers) == self.batch_size:
					batch_loss = self._train_batch(
						centers,
						contexts,
						input_embed,
						output_embed,
						optimizer,
						neg_probs,
						device,
					)
					total_loss += batch_loss
					batches += 1
					if progress_bar is not None:
						progress_bar.set_postfix(loss=f"{(total_loss / batches):.4f}")
					centers = [] # reset for next batch
					contexts = [] # reset for next batch

			# Train on any remaining pairs in the last batch
			if centers:
				batch_loss = self._train_batch(
					centers,
					contexts,
					input_embed,
					output_embed,
					optimizer,
					neg_probs,
					device,
				)
				total_loss += batch_loss
				batches += 1
				if progress_bar is not None:
					progress_bar.set_postfix(loss=f"{(total_loss / batches):.4f}")

			epoch_loss = total_loss / max(1, batches)
			print(f"Epoch {epoch + 1}/{self.epochs} - loss: {epoch_loss:.4f}")

		self.input_embedding_table = input_embed.weight.detach().cpu()
		self.output_embedding_table = output_embed.weight.detach().cpu()
		vectors = self.input_embedding_table.numpy().astype(np.float32)
		norms = np.linalg.norm(vectors, axis=1, keepdims=True)
		norms = np.where(norms == 0.0, 1.0, norms)
		self.embeddings = vectors / norms

	def get_vector(self, token):
		if self.embeddings is None:
			raise ValueError("Model has not been trained or loaded")
		if token not in self.vocab_index:
			raise ValueError(f"Token '{token}' not in vocabulary")
		return self.embeddings[self.vocab_index[token]]

	def most_similar(self, query, topn=10):
		if self.embeddings is None:
			raise ValueError("Model has not been trained or loaded")
		if topn <= 0:
			return [], []

		exclude_index = None
		if isinstance(query, str):
			if query not in self.vocab_index:
				raise ValueError(f"Token '{query}' not in vocabulary")
			exclude_index = self.vocab_index[query]
			query_vector = self.embeddings[exclude_index]
		else: # If the query is an embedding vector rather than a token.
			query_vector = np.asarray(query, dtype=np.float32)
			if query_vector.ndim != 1 or query_vector.shape[0] != self.embedding_dim:
				raise ValueError(f"Query vector must have shape ({self.embedding_dim},)")
			norm = float(np.linalg.norm(query_vector))
			if norm == 0.0:
				raise ValueError("Query vector norm must be > 0")
			query_vector = query_vector / norm

		similarities = self.embeddings @ query_vector
		if exclude_index is not None:
			similarities[exclude_index] = -np.inf

		max_candidates = len(self.vocab_index) - (1 if exclude_index is not None else 0)
		topn = min(topn, max_candidates)
		if topn <= 0:
			return [], []

		top_indices = np.argpartition(similarities, -topn)[-topn:]
		top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

		tokens = [self.index_vocab[int(idx)] for idx in top_indices]
		scores = [float(similarities[int(idx)]) for idx in top_indices]
		return tokens, scores

	def save(self, path):
		if self.embeddings is None:
			raise ValueError("Model has not been trained or loaded")
		if self.negative_sampling_probs is None:
			raise ValueError("Negative sampling probabilities are not initialized")

		payload = {
			"embedding_dim": self.embedding_dim,
			"window_size": self.window_size,
			"num_negatives": self.num_negatives,
			"epochs": self.epochs,
			"batch_size": self.batch_size,
			"learning_rate": self.learning_rate,
			"vocab_index": self.vocab_index,
			"index_vocab": self.index_vocab,
			"token_counts": self.token_counts,
			"negative_sampling_probs": self.negative_sampling_probs.cpu(),
			"embeddings": self.embeddings,
			"input_embedding_table": self.input_embedding_table,
			"output_embedding_table": self.output_embedding_table,
		}
		torch.save(payload, path)
		print(f"Model saved to {path}")

	@classmethod
	def load(cls, path):
		data = torch.load(path, weights_only=False)
		model = cls(
			embedding_dim=data["embedding_dim"],
			window_size=data["window_size"],
			num_negatives=data["num_negatives"],
			epochs=data["epochs"],
			batch_size=data["batch_size"],
			learning_rate=data["learning_rate"],
		)

		model.vocab_index = data["vocab_index"]
		model.index_vocab = data["index_vocab"]
		model.token_counts = data.get("token_counts")
		model.negative_sampling_probs = data.get("negative_sampling_probs")
		model.embeddings = data["embeddings"]
		model.input_embedding_table = data.get("input_embedding_table")
		model.output_embedding_table = data.get("output_embedding_table")
		print(f"Model loaded from {path}")
		return model

	def vocab(self):
		return self.vocab_index.keys()

	def dimension(self):
		return self.embedding_dim

	def nearest_neighbors(self, vector, topn=10):
		raise NotImplementedError

if __name__ == "__main__":
	# Example usage
	corpus = [[token.lower() for token in sentence] for sentence in brown.sents()]
	# model = Word2VecEmbedding(
	# 	embedding_dim=100,
	# 	window_size=2,
	# 	num_negatives=5,
	# 	epochs=5,
	# 	batch_size=2048,
	# 	learning_rate=0.01,
	# )
	# model.train(corpus, show_progress=True, device="cuda" if torch.cuda.is_available() else "cpu")
	# model.save(f"./embeddings/word2vec{model.window_size}.pt")
	model = Word2VecEmbedding.load("./embeddings/word2vec2.pt")

	query = "woman" if "woman" in model.vocab_index else next(iter(model.vocab_index))
	tokens, scores = model.most_similar(query, topn=10)
	print(f"Most similar to '{query}':")
	for token, score in zip(tokens, scores):
		print(f"Token: {token}, Similarity: {score:.4f}")

	def analogy_top5(a, b, c, candidates=50, topn=5):
		missing = [word for word in (a, b, c) if word not in model.vocab_index]
		if missing:
			return missing, []

		vector = model.get_vector(b) - model.get_vector(a) + model.get_vector(c)
		vector = vector / (np.linalg.norm(vector) + 1e-12)
		candidate_tokens, candidate_scores = model.most_similar(vector, topn=candidates)
		blocked = {a, b, c}
		filtered = [(token, score) for token, score in zip(candidate_tokens, candidate_scores) if token not in blocked]
		return [], filtered[:topn]

	analogy_cases = [
		("1. Paris : France :: Delhi : ? (Syntactic/Capital)", "paris", "france", "delhi"),
		("2. King : Man :: Queen : ? (Semantic/Gender)", "king", "man", "queen"),
		("3. Swim : Swimming :: Run : ? (Syntactic/Tense)", "swim", "swimming", "run"),
	]

	print("\nTop 5 analogy predictions:")
	for label, a, b, c in analogy_cases:
		missing_words, results = analogy_top5(a, b, c, candidates=50, topn=5)
		print(label)
		if missing_words:
			print("Missing words in vocabulary:", ", ".join(missing_words))
		else:
			for token, score in results:
				print(f"  {token}\t{score:.4f}")
		print("---")