"""Assembly of the data set D9 from data/raw/d9: the parsers of the individual sources, then build.py.

  belstat_prices   Belstat monthly average prices                     -> by_belstat_prices.csv
  nbrb_rates       NBRB interest rates (fetch_blocked.sh)              -> by_nbrb_rates.csv
  nbrb_bulletin    NBRB statistical bulletins (fetch_blocked.sh)       -> by_nbrb_bulletin.csv
  minfin_register  Belarusian bonds in circulation                     -> by_bank_bonds.csv
  pl_retail_bonds  Polish savings bonds, terms of each issue           -> pl_retail_bonds_issues.csv
  pl_bond_yields   realized yields of those issues                     -> pl_retail_bonds_yields.csv
  build            all series of the four countries                    -> monthly_long.csv, monthly_wide_*.csv, sources.csv
A parser whose input is missing (NBRB files before fetch_blocked.sh) is skipped; build.py then leaves its series out.
Run (from code/):  python -m lab10_economies.collect.build_all
"""
import runpy
import sys

from ..paths import RAW

STEPS = [("belstat_prices", []), ("nbrb_rates", []), ("nbrb_bulletin", []), ("minfin_register", []),
         ("pl_retail_bonds", ["parse"]), ("pl_bond_yields", []), ("build", [])]


def main():
    for name, args in STEPS:
        if name.startswith("nbrb") and not (RAW / "manual" / "nbrb").exists():
            print(f"skip {name}: no NBRB files (run collect/fetch_blocked.sh from a Belarusian connection)")
            continue
        print(f"--- {name}", flush=True)
        sys.argv = [name, *args]
        runpy.run_module(f"lab10_economies.collect.{name}", run_name="__main__")


if __name__ == "__main__":
    main()
