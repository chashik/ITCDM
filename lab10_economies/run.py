"""Lab 10: all stages on the data set D9 (about six minutes, CPU only); the printed report of each stage goes to
results/lab10/<stage>_report.txt.

  prepare      analysis panels of the four countries (data/processed/d9/panel_*.csv)
  growth       price growth against official inflation, rates and yields; same-unit table        (stage q1)
  relations    related series, nowcasts from the others, Granger tests, lead and lag              (stage q2)
  patterns     price setting, the inflation episode, regimes, pass-through, cross-country timing  (stage q3)
  forecasting  simple against advanced forecasting models; mutual information                    (stage q4)
  compare      the four countries in one table                                                    (stage q5)
  fuzzy_tools  fuzzy match, linguistic summaries, Takagi-Sugeno reaction rules, fuzzy regimes     (stage q6)
  further      the further methods of chapter 4: k-medoids, spectral clustering, hidden Markov model, FP-growth,
               conditional inference trees, sequential covering, Bayesian network                 (stage q7)
  illustrations  figures that accompany the tables of stages 1-5 (paper/figures/lab10_*)
  report       tables and numbers for Part 3 of the guide (paper/generated/lab10_*)
Run (from code/):  python -m lab10_economies.run
"""
import contextlib
import importlib
import io
import warnings

from .paths import RESULTS

STAGES = ["prepare", "growth", "relations", "patterns", "forecasting", "compare", "fuzzy_tools", "further", "illustrations", "report"]


def main():
    for name in STAGES:
        print(f"--- {name}", flush=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module(f"lab10_economies.{name}").main()
        (RESULTS / f"{name}_report.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main()
