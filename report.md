2.1 - Analogy evaluation

==== SVD (svd.pt) ====
Top 5 analogy predictions:
1. paris : france :: delhi : ? (Syntactic/Capital)
  confederacy   0.8383
  cathedral     0.8351
  mexico        0.8317
  handkerchief  0.8260
  manure        0.8254
---
2. king : man :: queen : ? (Semantic/Gender)
  woman 0.7422
  boy   0.7321
  still 0.7078
  music 0.7057
  young 0.7053
---
3. swim : swimming :: run : ? (Syntactic/Tense)
  floating      0.8182
  ran   0.8139
  climbed       0.8065
  flying        0.8035
  rode  0.7912
---

==== Word2Vec (word2vec2.pt) ====
Top 5 analogy predictions:
1. paris : france :: delhi : ? (Syntactic/Capital)
  fibrosis      0.4555
  boiled        0.4359
  acrylic       0.4296
  egypt 0.4258
  yarns 0.4160
---
2. king : man :: queen : ? (Semantic/Gender)
  girl  0.4508
  woman 0.4478
  writer        0.4372
  immaculate    0.4336
  leg   0.4139
---
3. swim : swimming :: run : ? (Syntactic/Tense)
  squeeze       0.4744
  cluster       0.4660
  wrinkles      0.4598
  flow  0.4483
  wax   0.4332
---
==== GloVe (glove-wiki-gigaword-100) ====
Top 5 analogy predictions:
1. paris : france :: delhi : ? (Syntactic/Capital)
  india 0.8602
  pakistan      0.7835
  lanka 0.6693
  bangladesh    0.6641
  sri   0.6440
---
2. king : man :: queen : ? (Semantic/Gender)
  woman 0.8040
  girl  0.7349
  she   0.6818
  her   0.6592
  mother        0.6542
---
3. swim : swimming :: run : ? (Syntactic/Tense)
  three 0.7016
  running       0.7008
  since 0.6961
  four  0.6906
  leading       0.6884
---

2.2 - Task 2 Bias check
==== GloVe (glove-wiki-gigaword-100) ====
2.2 Task 2 - Pairwise cosine similarity bias check
doctor: cos(doctor, man) = 0.6092, cos(doctor, woman) = 0.6333
nurse: cos(nurse, man) = 0.4562, cos(nurse, woman) = 0.6139
homemaker: cos(homemaker, man) = 0.2356, cos(homemaker, woman) = 0.4258
---