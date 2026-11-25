# Code of the guide «Introduction to Computer Data Mining»

Python implementation of the guide, public data only:

* **Part 2**, Labs 0–9: financial markets and consumer credit (FRED, FRED-MD, Yahoo Finance, Kenneth R. French Data
  Library, SEC EDGAR, Federal Reserve FOMC calendars, UCI Machine Learning Repository, financial news tweets);
* **Part 3**, chapter 4: the further methods of data mining and their worked examples (`further_methods/`: k-medoids,
  spectral clustering, hidden Markov model, FP-growth, conditional inference trees, sequential covering, Bayesian networks);
* **Part 3**, Lab 10: prices, inflation and interest rates of Belarus, Poland, Russia and Lithuania, 2021–2025 (data set
  D9: national statistical offices and central banks, the EEC, FAO, Eurostat, OECD, IMF, BIS, EU Oil Bulletin, finance
  ministries, Moscow Exchange);
* **Part 1**, chapter 2 (mathematical foundations): the numerical checks of its statements on the data of the
  labs (`math_foundations/`), and the illustrations of the theory (`theory_figs*.py`).

```sh
make venv      # create .venv and install the pinned requirements
export DMFIN_SEC_CONTACT='Your Name your.email@example.org'   # required by SEC EDGAR, see "Data" below
make data      # download and cache the data of Labs 0-9 (data/raw, data/processed), compare them with data_manifest.csv
make data-d9   # download and assemble the data set D9 of Lab 10 (see "Data of Lab 10" below)
make test      # 139 unit tests, offline, synthetic data (about a minute)
make lab0      # ... lab10, math, further, theory: one experiment; figures -> ../paper/figures, tables and numbers -> ../paper/generated
make all       # data, data-d9, test and every experiment in order
make dist      # ../dmfin-code.zip: the code with results, without environment, data and caches
```

## Licence

The code is distributed under the MIT License (`LICENSE`). The licence covers the code only: the data are not
part of this package and remain subject to the terms of their providers.

## Data

The data are not redistributed with the code: several providers do not permit it (Yahoo Finance; series of FRED
owned by third parties, such as Moody's yields or Cboe's VIX). `make data` downloads them from the providers into
`data/raw` (289 files, about 84 MB; with D9, 2 889 files and about 930 MB). The SEC asks every automated client to declare its own contact, so
the EDGAR company data are downloaded only when the environment variable `DMFIN_SEC_CONTACT` holds your name and
e-mail address; the files are cached, so the variable is needed only for the first download.

Providers revise their data — adjusted prices, revised series, new observations — so a new download can differ
from the one on which the numbers of the guide were computed (October 2026). `data_manifest.csv` records, for
every file of that download, its size, number of lines, first and last date and SHA-256 checksum; `make
check-data` (run automatically by `make data`) lists the files that differ. FRED files always contain the latest
observations and therefore show as changed; the code uses data up to 30 June 2026 only. If results differ from
the guide, compare them with the reference values in `results/*.json` and `results/*.csv` and look first at
the files reported as changed.

### Data of Lab 10 (D9)

`make data-d9` downloads the files of every source that answers automated requests into `data/raw/d9`
(2 600 files, about 850 MB, mostly the PDF bulletins of the National Bank of Belarus) and assembles the monthly
series in `data/processed/d9` (`lab10_economies/collect/build_all.py`). Three sites answer only from a Belarusian
connection (National Bank of Belarus, Belarusian Currency and Stock Exchange, Rosstat): `collect/fetch_blocked.sh`
downloads their files once, by hand. Rules of the collection:

* downloaded files are untrusted data: each source has its own folder, the files are only parsed, never executed;
* each indicator is taken from its producer (the central bank for its rates, the statistical office for its prices
  and inflation); the other sources are kept in `monthly_long.csv` as checks and are never averaged with it;
* policy rates are the mean over the days of the month of the rate in force; the consumer price index is chained
  from the official monthly changes and benchmarked to the published December rates;
* a value is accepted only after a comparison with a second source or with what is plausible; the two errors that
  this found (a parser, a republication) are described in Lab 10, subsection 5.1.3 of Part 3.

| Folder | Laboratory assignment |
|---|---|
| `dmfin/` | shared layer: configuration and chronological split, data download (`data.py`, `data_ext.py`) and its manifest (`manifest.py`), panel of 62 series (`panel.py`), features, splits, plotting style, export to LaTeX |
| `lab0_domain/` | 0. Research domain, data inventory, data quality, stylised facts; `run_ext.py`: further sources, coverage, panel |
| `lab1_clustering/` | 1. Exploratory analysis of 110 stocks with clustering (Hopkins, elbow, silhouette, hierarchical, DBSCAN, stability, out-of-sample check); `run_ext.py`: Gaussian mixtures, HDBSCAN, t-SNE, 49 industries, market regimes |
| `lab2_regression/` | 2. Polynomial regression of next-month volatility (normalisation, PCA, CV with embargo); `run_ext.py`: FRED-MD, multiple testing, conformal intervals, FOMC event study, Granger causality |
| `lab3_classification/` | 3. Credit default: logistic regression, perceptrons, trees (normalisation, PCA, cost-sensitive thresholds); `run_ext.py`: multilayer perceptron study, SVM, kNN, LDA, naive Bayes, German credit; `run_text.py`: sentiment of financial news tweets — naive Bayes and TF-IDF against an LSTM and DistilRoBERTa (random weights, fixed features, fine-tuned; GPU) |
| `lab4_trees/` | 4. Decision trees and ensembles: pruning, random forests, boosting, importance and SHAP, regression trees, a global boosting model |
| `lab5_fuzzy/` | 5. Fuzzy c-means, Mamdani stress index, Wang–Mendel fuzzy rules; `run_ext.py`: Apriori association rules |
| `lab6_rnn/` | 6. RNN/LSTM/GRU: forecasting, classification, anomaly detection, motif discovery, error-correcting network; `run_mp.py`: matrix profile, Isolation Forest; `run_global.py`: global LSTM/GRU/TCN/Transformer on 62 series, Chronos-Bolt and Chronos-2 (with VIX as a covariate), conformal intervals |
| `lab7_rl_features/` | 7. Q-learning feature selection on a volatile stream stored in a document database (TinyDB); `run_ext.py`: 50 features, SARSA, LinUCB, DQN, unlabeled variant |
| `lab8_genetic/` | 8. Genetic algorithms for the networks of Lab 6 and the feature selection of Lab 7; `run_search.py`: GA vs random search, TPE, Hyperband, PBT; `run_features_ext.py`: GA on 50 features |
| `lab9_kg/` | 9. Knowledge graph of the domain: ontology, OWL 2 RL reasoning, SHACL, SPARQL, graph mining |
| `lab10_economies/` | 10. Four economies, 2021–2025: `collect/` (data set D9), `prepare.py` (panels), seven stages — growth and real rates in one unit, government bonds as yield and as holder's return (`growth.py`), relations and Granger tests (`relations.py`), price setting, regimes and pass-through (`patterns.py`), fuzzy match, linguistic summaries, Takagi–Sugeno rules, fuzzy c-means (`fuzzy_tools.py`), forecasting with Diebold–Mariano tests and mutual information (`forecasting.py`), the further methods of chapter 4 (`further.py`), comparison (`compare.py`); `illustrations.py`, `report.py` |
| `further_methods/` | Chapter 4 of Part 3: PAM, PAMk and CLARA (`kmedoids.py`), normalized spectral clustering (`spectral.py`), Gaussian hidden Markov model with Baum–Welch and Viterbi (`hmm.py`), FP-growth (`fpgrowth.py`), conditional inference trees (`ctree.py`), sequential covering with FOIL gain (`covering.py`), discrete Bayesian networks with hill climbing on BIC/AIC (`bayesnet.py`); worked examples (`examples.py`); applied in stage 6 of Lab 10 (`lab10_economies/further.py`) |
| `math_foundations/` | Chapter 2 of Part 1: linear algebra basics (`la_basics.py`: shapes of the tensors of the labs, correlation as a cosine, portfolio variance, PCA by the spectral theorem, normal equations, backpropagation as the chain rule) and the checks of sections 2.2–2.11 (`m0_boxes.py` … `m9_fixed_points.py`) |
| `theory_figs.py`, `theory_figs_ext.py` | illustrations of the topics of Part 1 |

## CPU and GPU

`make venv` installs the CPU build of PyTorch, with which every experiment runs. A GPU is optional: to use one,
install the PyTorch build that matches the GPU and its driver (see https://pytorch.org/get-started/locally/).
A GPU is used in three places only: the genetic search of Lab 8.1 for networks with at least 64 hidden units (a
recurrent network is trained step by step along the sequence, and a GPU shortens the training of small networks
little), the global networks and the pretrained forecasters of Lab 6 (`run_global.py`) and the text networks of
Lab 3 (`run_text.py`: about 45 minutes with a GPU, several hours on the CPU alone); the rule is in
`dmfin/device.py`, the measurement behind it in `python -m dmfin.bench_devices` (`make bench-devices`).
`DMFIN_DEVICE=cpu` disables the GPU. CPU-heavy work and GPU work run one after the other: all cores at full load
together with the GPU switched off the author's computer (power supply); `DMFIN_OVERLAP=1` allows both at once.
Each device is deterministic for a fixed seed; CPU and GPU results differ in the last digits only (Lab 6,
`run_devices.py`), except for networks with dropout, whose random masks the GPU draws from its own generator.

## Numbers in the text

Every number quoted in the guide (all three parts) is written by the code into
`../paper/generated/*_numbers.tex`, and every generated table into `../paper/generated/*.tex`;
`python -m dmfin.reexport` rebuilds the number files from `results/*_numbers.json` without recomputation.
The text contains the macros, never the numbers themselves, so a rerun changes the text consistently.
