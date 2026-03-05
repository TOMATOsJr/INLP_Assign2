==== SVD (svd.pt) ====
Top 5 analogy predictions:
1. paris : france :: delhi : ? (Syntactic/Capital)
  1789  0.6099
  vigilant      0.5907
  nameless      0.5880
  combatant     0.5808
  scripture     0.5645
---
2. king : man :: queen : ? (Semantic/Gender)
  woman 0.6936
  boy   0.6582
  young 0.6547
  girl  0.6475
  student       0.5865
---
3. swim : swimming :: run : ? (Syntactic/Tense)
  flying        0.7462
  ran   0.6922
  spread        0.6769
  boat  0.6735
  running       0.6732
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