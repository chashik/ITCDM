"""Chapter 2 of Part 1 (mathematical foundations): all numerical examples (about four minutes, CPU only; needs the results of Labs 1 and 10 and the data of
Labs 2, 3, 5).  The printed report of each chapter goes to results/math/<module>_report.txt.
Run (from code/):  python -m math_foundations.run
"""
import contextlib
import importlib
import io
import warnings

from dmfin import config as C

MODULES = ["la_basics", "m0_boxes", "m1_linear_algebra", "m2_optimization", "m3_semirings", "m4_lattices", "m5_fuzzy_algebra",
           "m6_invariance", "m7_index_numbers", "m8_algebraic_statistics", "m9_fixed_points"]


def main():
    out = C.RESULTS_DIR / "math"
    out.mkdir(exist_ok=True)
    for name in MODULES:
        print(f"--- {name}", flush=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            importlib.import_module(f"math_foundations.{name}").main()
        (out / f"{name}_report.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main()
