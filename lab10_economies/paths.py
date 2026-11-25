"""Paths of Lab 10: the data set D9 (four economies) and the outputs of the stages."""
from dmfin import config as C

RAW = C.RAW_DIR / "d9"              # downloads, treated as untrusted (collect/download.py, collect/fetch_blocked.sh)
DATA = C.PROC_DIR / "d9"            # collected tables (monthly_long.csv, monthly_wide_*.csv, sources.csv) and panels
RESULTS = C.RESULTS_DIR / "lab10"   # tables (CSV) and printed reports of the stages
for _d in (RAW, DATA, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)
