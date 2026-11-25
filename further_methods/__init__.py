"""Part 3, chapter 4 (further methods of data mining): implementations of the methods that Part 1 does not treat, used
by the worked examples of the chapter (``examples.py``) and by stage 6 of Lab 10 (``lab10_economies.further``).

  kmedoids   PAM (BUILD and SWAP), PAMk (number of clusters by the average silhouette), CLARA (PAM on samples)
  spectral   normalized spectral clustering (Ng, Jordan and Weiss) and the eigengap heuristic
  hmm        Gaussian hidden Markov model: forward-backward with scaling, Baum-Welch, Viterbi
  fpgrowth   FP-tree and FP-growth (frequent itemsets without candidate generation)
  ctree      conditional inference trees (Hothorn, Hornik and Zeileis): permutation-test variable selection
  covering   rule induction by sequential covering with FOIL gain
  bayesnet   discrete Bayesian networks: BIC score, hill climbing, inference by enumeration
"""
