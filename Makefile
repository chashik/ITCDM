# Reproduce every experiment, figure and table of Part 2 of the guide.
PY = .venv/bin/python

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

bench-devices:        ## time of the PyTorch workloads of the labs on CPU and GPU (table in the guide)
	$(PY) -m dmfin.bench_devices

gpu-check:            ## is a CUDA/ROCm device usable for the recurrent networks?
	$(PY) -c "import torch; from lab6_rnn.nets import gpu_rnn_check; \
	print(torch.__version__, torch.cuda.is_available() and torch.cuda.get_device_name(0)); \
	print('RNN with dropout on the GPU:', (gpu_rnn_check() or 'OK') if torch.cuda.is_available() else 'no GPU')"

data:                 ## download and cache the public data (needs Internet once and DMFIN_SEC_CONTACT, see README)
	$(PY) -m dmfin.data
	$(PY) -m dmfin.data_ext
	-$(PY) -m dmfin.manifest --check

check-data:           ## compare data/raw with data_manifest.csv, the data on which the guide was computed
	$(PY) -m dmfin.manifest --check

data-d9:              ## Lab 10: download and assemble the four-economy data set D9 (NBRB/BCSE/Rosstat: collect/fetch_blocked.sh)
	$(PY) -m lab10_economies.collect.download
	$(PY) -m lab10_economies.collect.build_all

manifest:             ## (author) record sizes, dates and checksums of data/raw in data_manifest.csv
	$(PY) -m dmfin.manifest

test:                 ## unit tests of all laboratory assignments (synthetic data, offline)
	$(PY) -m pytest

lab0:
	$(PY) -m lab0_domain.run
	$(PY) -m lab0_domain.run_ext
lab1:
	$(PY) -m lab1_clustering.run
	$(PY) -m lab1_clustering.run_ext
lab2:
	$(PY) -m lab2_regression.run
	$(PY) -m lab2_regression.run_ext
lab3:
	$(PY) -m lab3_classification.run
	$(PY) -m lab3_classification.run_ext
	$(PY) -m lab3_classification.run_text     # text classifiers; GPU if there is one
lab4: ; $(PY) -m lab4_trees.run
lab5:
	$(PY) -m lab5_fuzzy.run
	$(PY) -m lab5_fuzzy.run_ext
lab6:
	$(PY) -m lab6_rnn.run_forecast
	$(PY) -m lab6_rnn.run_classify
	$(PY) -m lab6_rnn.run_anomaly
	$(PY) -m lab6_rnn.run_motifs
	$(PY) -m lab6_rnn.run_adaptive
	$(PY) -m lab6_rnn.run_devices      # needs a GPU, otherwise skipped
	$(PY) -m lab6_rnn.run_global
	$(PY) -m lab6_rnn.run_mp
lab7:
	$(PY) -m lab7_rl_features.run
	$(PY) -m lab7_rl_features.run_ext
lab8:
	$(PY) -m lab8_genetic.run_lstm
	$(PY) -m lab8_genetic.run_features
	$(PY) -m lab8_genetic.run_search
	$(PY) -m lab8_genetic.run_features_ext
lab9: ; $(PY) -m lab9_kg.run
lab10: ; $(PY) -m lab10_economies.run
math: ; $(PY) -m math_foundations.run       ## Part 1, chapter 2 (mathematical foundations): numerical examples (after lab1 and lab10)
further: ; $(PY) -m further_methods.examples   ## Part 3, chapter 4 (further methods): worked examples
theory:
	$(PY) theory_figs.py
	$(PY) theory_figs_ext.py

all: data data-d9 test lab0 lab1 lab2 lab3 lab4 lab5 lab6 lab7 lab8 lab9 lab10 math further theory

dist:                 ## zip archive of the code to be shared with the guide: no environment, data, caches or logs
	cd .. && rm -f dmfin-code.zip && zip -rqX dmfin-code.zip code \
	  -x 'code/.git/*' 'code/.venv/*' 'code/data/*' '*/__pycache__/*' '*/.pytest_cache/*' \
	  'code/results/*_cache/*' '*.pkl' '*.pt' '*.parquet' '*.log' \
	  'code/results/lab7_stream.json'
	@echo "../dmfin-code.zip"

.PHONY: venv gpu-check bench-devices data data-d9 check-data manifest test all dist lab0 lab1 lab2 lab3 lab4 lab5 lab6 lab7 lab8 lab9 lab10 math further theory
