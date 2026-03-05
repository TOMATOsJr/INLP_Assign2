# Assignment 2 Report: Word Embeddings and Analysis

## 1.1 SVD-Based Word Embeddings

### Method
I implemented a count-based embedding pipeline with the following steps:

1. Corpus preparation: Brown corpus sentences from NLTK, lowercased.
2. Co-occurrence matrix construction: for each center word, context words in a symmetric window are counted.
3. Context weighting: inverse-distance weighting `w = 1 / distance` inside the window.
4. PMI transformation: raw co-occurrence counts are converted to Positive PMI (PPMI).
5. Dimensionality reduction: `TruncatedSVD` is applied on the sparse PPMI matrix.
6. L2 normalization: vectors are normalized before cosine-similarity-based evaluation.

### Hyperparameters and Justification

Model checkpoint used: `svd.pt`

- Embedding dimensionality: `n_components = 100`
- Context window size: `window_size = 2`
- Weighting scheme: inverse-distance (`1/distance`)
- Matrix transform: PPMI
- SVD variant: Truncated SVD with `random_state = 42`

Justification:

1. `window_size = 2` prioritizes local syntactic structure, which is appropriate for analogy tests containing morphological and relation-level patterns.
2. Inverse-distance weighting gives stronger contribution to nearer words and reduces noise from farther context tokens, while still using information from both sides.
3. PPMI suppresses very frequent but uninformative co-occurrences and emphasizes unexpectedly informative word pairs.
4. `100` dimensions is a practical tradeoff between expressiveness and overfitting/noise, and also keeps comparison fair with the pre-trained 100d GloVe vectors.
5. Truncated SVD is computationally suitable for large sparse matrices and preserves major latent semantic factors.

Observed vocabulary size in checkpoint: `49,815`.

## 1.2 Neural Word Embedding (Word2Vec)

### Method
I implemented Skip-Gram with Negative Sampling (SGNS) in PyTorch.

Given a center word, the model predicts surrounding context words. For each positive `(center, context)` pair, it samples negative words and optimizes:

`log sigma(u_context^T v_center) + sum_i log sigma(-u_neg_i^T v_center)`

Implementation details:

1. Vocabulary filtering using minimum frequency.
2. Skip-gram pair generation using symmetric context window.
3. Negative sampling distribution proportional to `count(w)^0.75`.
4. Two embedding tables (input/output) trained with mini-batches.
5. Final exported vectors are L2-normalized input embeddings.

### Hyperparameters and Justification

I evaluated two Word2Vec runs so the effect of vocabulary pruning can be compared directly.

Shared settings in both runs:

- Architecture: Skip-Gram with Negative Sampling
- Embedding dimensionality: `embedding_dim = 100`
- Context window size: `window_size = 2`
- Negative samples per positive pair: `num_negatives = 5`
- Epochs: `8`
- Batch size: `2048`
- Optimizer: Adam
- Learning rate: `0.003`

Variable setting:

- Run 1 (original): `min_freq = 5`
- Run 2 (new): `min_freq = 2`

Justification:

1. Skip-gram is typically stronger than CBOW for learning good vectors for less frequent words.
2. `100` dimensions again gives fair cross-model comparison and manageable training cost.
3. `window_size = 2` aligns with the SVD setup for controlled comparison.
4. `num_negatives = 5` is a standard SGNS setting that balances gradient quality and speed.
5. `min_freq` controls vocabulary pruning. `5` keeps only more frequent tokens (less noise), while `2` includes more low-frequency words (richer coverage but potentially noisier neighbors).
6. `epochs = 8` allows repeated exposure to corpus co-occurrences without excessively long training.
7. Adam with `lr = 0.003` and large batches (`2048`) provides stable optimization on this dataset size.

Observed vocabulary size for the `min_freq = 2` checkpoint (`word2vec2.pt`): `27,805`.

## 2. Analysis: Are the Embeddings Fishy?

## 2.1 Task 1: Analogy Test (Semantic Capability)

Analogy query vector:

`q = vec(B) - vec(A) + vec(C)`

Prediction rule:

`argmax_x cos(x, q)`

Top-5 predictions were computed for SVD, Word2Vec, and pre-trained GloVe (`glove-wiki-gigaword-100`).

### A) SVD (`svd.pt`)

1. `paris : france :: delhi : ?`
- `1789` (0.6099)
- `vigilant` (0.5907)
- `nameless` (0.5880)
- `combatant` (0.5808)
- `scripture` (0.5645)

2. `king : man :: queen : ?`
- `woman` (0.6936)
- `boy` (0.6582)
- `young` (0.6547)
- `girl` (0.6475)
- `student` (0.5865)

3. `swim : swimming :: run : ?`
- `flying` (0.7462)
- `ran` (0.6922)
- `spread` (0.6769)
- `boat` (0.6735)
- `running` (0.6732)

Interpretation:

1. SVD captures some gender relation structure (`woman` ranked first).
2. Capital-country relation is weak in this trained space (no `india` in top results).
3. Verb morphology is partially encoded (`ran`, `running` appear) but noisy terms outrank expected forms.
4. High-vocabulary, no frequency filtering in SVD likely introduces noisy rare-token effects.

### B) Word2Vec (Two Runs, `window_size=2`)

Run 1: `min_freq = 5` (original run)

1. `paris : france :: delhi : ?`
- `fibrosis` (0.4555)
- `boiled` (0.4359)
- `acrylic` (0.4296)
- `egypt` (0.4258)
- `yarns` (0.4160)

2. `king : man :: queen : ?`
- `girl` (0.4508)
- `woman` (0.4478)
- `writer` (0.4372)
- `immaculate` (0.4336)
- `leg` (0.4139)

3. `swim : swimming :: run : ?`
- `squeeze` (0.4744)
- `cluster` (0.4660)
- `wrinkles` (0.4598)
- `flow` (0.4483)
- `wax` (0.4332)

Run 2: `min_freq = 2` (new run, `word2vec2.pt`)

1. `paris : france :: delhi : ?`
- `materialism` (0.4577)
- `1861` (0.4526)
- `whereof` (0.4514)
- `backbends` (0.4488)
- `duplicated` (0.4328)

2. `king : man :: queen : ?`
- `housewife` (0.4555)
- `affair` (0.4225)
- `communion` (0.4145)
- `preacher` (0.4077)
- `pitchfork` (0.4072)

3. `swim : swimming :: run : ?`
- `roundabout` (0.4736)
- `strip` (0.4345)
- `clamping` (0.4250)
- `terrific` (0.4101)
- `short-term` (0.4093)

Interpretation:

1. Both Word2Vec runs underperform on the three analogy prompts compared to GloVe and do not consistently recover expected answers.
2. Changing `min_freq` from `5` to `2` noticeably changes nearest-neighbor structure, showing sensitivity to vocabulary pruning.
3. The `min_freq = 2` run increases lexical coverage but also introduces more noisy low-frequency tokens among top predictions.
4. Overall behavior suggests corpus size/domain and training signal are the main bottlenecks, while `min_freq` mainly shifts the noise-coverage tradeoff.

### C) Pre-trained GloVe (`glove-wiki-gigaword-100`)

1. `paris : france :: delhi : ?`
- `india` (0.8602)
- `pakistan` (0.7835)
- `lanka` (0.6693)
- `bangladesh` (0.6641)
- `sri` (0.6440)

2. `king : man :: queen : ?`
- `woman` (0.8040)
- `girl` (0.7349)
- `she` (0.6818)
- `her` (0.6592)
- `mother` (0.6542)

3. `swim : swimming :: run : ?`
- `three` (0.7016)
- `running` (0.7008)
- `since` (0.6961)
- `four` (0.6906)
- `leading` (0.6884)

Interpretation:

1. GloVe is strongest overall, correctly returning `india` and `woman` at rank 1.
2. The tense-style query is mixed: `running` is near top, but unrelated numerals appear above/near it, showing analogy formulation can still be brittle.
3. Larger and more diverse pre-training corpus gives better relational geometry than task-trained local models.

### Overall Comparison for Task 2.1

1. Best semantic performance: GloVe.
2. Moderate performance: SVD (good on gender, weak on geography, mixed morphology).
3. Weakest in this setup: trained Word2Vec.
4. Result supports the known pattern that high-resource pre-trained embeddings outperform small-corpus custom training on analogy benchmarks.

## 2.2 Task 2: Bias Check (Pre-trained Embeddings Only)

Model used: GloVe (`glove-wiki-gigaword-100`)

Pairwise cosine similarities:

1. `doctor`
- `cos(doctor, man) = 0.6092`
- `cos(doctor, woman) = 0.6333`

2. `nurse`
- `cos(nurse, man) = 0.4562`
- `cos(nurse, woman) = 0.6139`

3. `homemaker`
- `cos(homemaker, man) = 0.2356`
- `cos(homemaker, woman) = 0.4258`

Interpretation:

1. `doctor` is slightly closer to `woman` than `man` in this embedding (`+0.0241` toward `woman`).
2. `nurse` shows a stronger association with `woman` (`+0.1577`).
3. `homemaker` is also notably closer to `woman` (`+0.1902`).
4. These directional differences indicate gendered structure in the embedding space.
5. This aligns with the broader finding that distributional embeddings reflect social patterns and stereotypes present in training text (Bolukbasi et al., 2016).

## 3. POS Tagging with MLP

### 3.4 Evaluation

Test-set performance for all embedding variants is reported below.

| Embedding Variant | Best Epoch | Best Validation Accuracy | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| SVD | 8 | 0.965133 | 0.963479 | 0.901470 |
| Word2Vec (`word2vec2`) | 8 | 0.965679 | 0.965152 | 0.918739 |
| GloVe (`glove-wiki-gigaword-100`) | 7 | 0.975385 | 0.974299 | 0.942424 |

Best-performing model: **GloVe**.

Confusion matrix for the best model (GloVe):

| True \ Pred | . | ADJ | ADP | ADV | CONJ | DET | NOUN | NUM | PRON | PRT | VERB | X |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| . | 14906 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| ADJ | 0 | 7654 | 4 | 91 | 0 | 0 | 473 | 1 | 0 | 3 | 76 | 0 |
| ADP | 6 | 1 | 14419 | 61 | 5 | 34 | 1 | 1 | 11 | 68 | 7 | 0 |
| ADV | 0 | 178 | 106 | 5257 | 12 | 21 | 53 | 0 | 0 | 27 | 31 | 0 |
| CONJ | 0 | 0 | 3 | 4 | 3762 | 9 | 0 | 0 | 0 | 0 | 2 | 0 |
| DET | 0 | 0 | 34 | 6 | 2 | 13632 | 3 | 0 | 16 | 2 | 0 | 0 |
| NOUN | 0 | 218 | 4 | 12 | 1 | 8 | 27006 | 29 | 2 | 34 | 274 | 13 |
| NUM | 0 | 1 | 0 | 0 | 0 | 0 | 32 | 1405 | 0 | 0 | 0 | 0 |
| PRON | 0 | 0 | 21 | 0 | 0 | 54 | 1 | 0 | 4899 | 1 | 0 | 0 |
| PRT | 0 | 0 | 132 | 4 | 0 | 0 | 58 | 0 | 1 | 2741 | 9 | 0 |
| VERB | 0 | 96 | 8 | 21 | 0 | 0 | 348 | 0 | 1 | 46 | 17774 | 3 |
| X | 1 | 3 | 0 | 0 | 0 | 1 | 39 | 0 | 0 | 0 | 1 | 65 |

![Confusion Matrix - GloVe POS Tagger](confusion_matrix.png)

Key observations from the confusion matrix:

1. Strong diagonal concentration confirms high overall performance.
2. Most common confusions are between syntactically related classes (for example `NOUN` vs `VERB`, `ADJ` vs `NOUN`, and `ADV` vs `ADJ/ADP`).
3. Minor classes such as `X` remain harder to classify, which is expected due to lower frequency and broader definition.

## 4. Analysis and Report

### 4.1 Embedding Comparison

For the POS tagging task, pre-trained GloVe **significantly outperformed** the embeddings trained from scratch.

1. GloVe vs SVD: `+1.08%` absolute test accuracy and `+0.04095` Macro-F1.
2. GloVe vs Word2Vec: `+0.91%` absolute test accuracy and `+0.02369` Macro-F1.
3. Word2Vec outperformed SVD on Macro-F1 (`0.918739` vs `0.901470`), indicating better balance across tag classes.

This supports the expectation that large-corpus pre-trained embeddings provide richer lexical semantics and better generalization for downstream sequence labeling.

### 4.2 MLP Hyperparameters

Hyperparameters used for the MLP POS tagger:

1. Context size `C = 1` (window size = `2C + 1 = 3` tokens).
2. Input dimensionality = `3 * embedding_dim` = `300` (all embeddings are 100d).
3. Hidden layers: `[256, 128]` with ReLU activations.
4. Dropout: `0.2`.
5. Optimizer: Adam.
6. Learning rate: `1e-3`.
7. Batch size: `256`.
8. Epochs: `8` (best checkpoint selected on validation accuracy).

### 4.3 Error Analysis (Selected Incorrect Predictions)

Using `error_examples_pos.json`, the following five errors were selected as requested: 3 from GloVe, 1 from SVD, and 1 from Word2Vec.

1. **SVD example**
Sentence: `2-2 .`
Mismatch: `2-2` gold=`NUM`, predicted=`NOUN`.
Analysis: Hyphenated numeric tokens are rare and ambiguous; the model likely maps this surface form to noun-like contexts instead of a numeric pattern.

2. **Word2Vec example**
Sentence: `A warning`
Mismatch: `warning` gold=`NOUN`, predicted=`VERB`.
Analysis: `warning` is lexically ambiguous (gerund/participle vs noun). With very short context, the model over-relies on word form and confuses nominal and verbal usage.

3. **GloVe example 1**
Sentence: `Contact`
Mismatch: `Contact` gold=`VERB`, predicted=`NOUN`.
Analysis: Single-word sentences remove contextual cues entirely. `contact` is highly ambiguous between noun and verb, and noun priors dominate.

4. **GloVe example 2**
Sentence: `Examples`
Mismatch: `Examples` gold=`NOUN`, predicted=`ADJ`.
Analysis: Sentence-initial capitalization and fragment-style input can weaken syntactic signals, causing adjective/noun confusion in short utterances.

5. **GloVe example 3**
Sentence: `Ranking .`
Mismatch: `Ranking` gold=`VERB`, predicted=`NOUN`.
Analysis: `-ing` forms are systematically ambiguous in English (verb vs nominalization). Without richer context, the model prefers the frequent noun reading.

Overall error pattern:

1. Most mistakes come from lexical ambiguity (noun/verb or noun/adjective boundary cases).
2. Very short or fragment sentences are harder because local context window (`C=1`) provides limited disambiguating evidence.
3. Unusual tokens (hyphenated numbers, proper names, edge punctuation patterns) increase out-of-distribution behavior.

## Discussion and Limitations

1. Analogy evaluation on only three questions is illustrative but not statistically strong.
2. Brown corpus size/domain is limited for training robust semantic vectors from scratch.
3. No subword modeling is used; rare/morphologically rich words are harder to represent.
4. SVD model uses full vocabulary without frequency cutoff, which can amplify rare-token noise.
5. The bias check reports association, not causation or social truth; interpretation must remain careful.

## Conclusion

1. The implemented SVD and SGNS pipelines are functional and produce meaningful vector spaces.
2. In this experiment, pre-trained GloVe clearly provides the strongest analogy performance and best POS tagging performance.
3. SVD retained some interpretable relations; custom Word2Vec underperformed on analogies but was competitive with SVD in POS tagging.
4. Bias auditing on GloVe reveals measurable gendered associations for profession terms.
5. Future improvements should include larger corpus training, stronger token filtering/subsampling, and broader intrinsic/extrinsic evaluations.

## Reference

Bolukbasi, T., Chang, K.-W., Zou, J. Y., Saligrama, V., and Kalai, A. T. (2016). *Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings*. NIPS.
