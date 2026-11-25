"""Chapter 2 of Part 1 of the guide: mathematical foundations of data mining — numerical examples on the guide's own data.

  m0_boxes          white, grey and black boxes: identifiability, parameter symmetries, Shapley values
  m1_linear_algebra least squares and conditioning, ridge through the SVD, PCA and the Eckart-Young theorem
  m2_optimization   gradient descent, SGD and Adam on logistic regression
  m3_semirings      exact segmentation as (min, +) matrix products, Mamdani inference as max-min composition, ReLU
                    networks as differences of convex piecewise-linear (tropical) functions
  m4_lattices       closed itemsets and the Galois connection; dendrograms and ultrametrics
  m5_fuzzy_algebra  t-norms, the failing law of excluded middle, monotone quantifiers
  m6_invariance     permutation tests and groups of transformations; equivariance of attention; seasonality on Z_12
  m7_index_numbers  price-index axioms, chain-linking and circularity; exact and approximate real rates
  m8_algebraic_stat exact tests on contingency tables with a Markov basis
  m9_fixed_points   contraction of the Bellman operator; Markov chains and Perron-Frobenius; rounding in chained indices
Each module writes paper/generated/mathN_numbers.tex (macros \\MathNXxx) and figures paper/figures/mathN_*.pdf.
Run (from code/):  python -m math_foundations.run   (or one module: python -m math_foundations.m3_semirings)
"""
