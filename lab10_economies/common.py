"""Shared helpers of Lab 10 (four economies): labels, panels, output of tables (results/lab10) and figures
(paper/figures/lab10_*.pdf, in the style of the guide)."""
import re
from pathlib import Path

import pandas as pd

from dmfin import plotting as P

from .paths import DATA, RESULTS

COUNTRIES = {"BY": "Belarus", "PL": "Poland", "RU": "Russia", "LT": "Lithuania"}
ALL = COUNTRIES
CURRENCY = {"BY": "BYN", "PL": "PLN", "RU": "RUB", "LT": "EUR"}
PERIOD = ("2021-01", "2025-12")             # study period; December 2020 (first row of a panel) is the base of growth
LABEL = {
    "flour_wheat": "wheat flour", "milk": "milk", "sugar": "sugar", "petrol": "petrol", "cpi_index": "consumer prices",
    "fx_usd": "US dollar rate", "cpi_yoy": "inflation (CPI, y/y)", "policy_rate": "policy rate",
    "mortgage_rate": "mortgage rate", "bond_gov": "government bond yield", "bond_gov_local": "gov. bond yield, local",
    "bond_bank": "bank bond coupon", "bond_retail": "retail bond (EDO) rate",
}
GOODS = ["flour_wheat", "milk", "sugar", "petrol"]


def panel(country: str, base: bool = False) -> pd.DataFrame:
    """The panel 2021-01..2025-12; base=True keeps the base row December 2020 in front (question 1)."""
    p = pd.read_csv(DATA / f"panel_{country}.csv", index_col=0)
    p.index = pd.PeriodIndex(p.index.astype(str), freq="M")
    return p if base else p.loc[PERIOD[0]:]


def save_table(df: pd.DataFrame, name: str, digits: int = 2) -> Path:
    path = RESULTS / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    df.round(digits).to_csv(path)
    return path


def save_figure(f, name: str):
    """PDF in paper/figures as lab10_<name> (stage prefixes q1_ ... q6_ dropped)."""
    return P.save(f, "lab10_" + re.sub(r"^q\d_", "", name))


def md_table(df: pd.DataFrame, digits: int = 1) -> str:
    """A small Markdown table for the report."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "–" if pd.isna(v) else f"{v:.{digits}f}")
    head = "| " + " | ".join([d.index.name or ""] + [str(c) for c in d.columns]) + " |"
    sep = "|" + "---|" * (len(d.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(d.index, d.values)]
    return "\n".join([head, sep] + rows)
