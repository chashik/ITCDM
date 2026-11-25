"""Lab 7. A volatile market-data stream stored in a document (NoSQL) database.

Each trading day becomes one JSON document in a TinyDB store:

    {"date": "2016-03-01", "src": "feed-B", "fields": {"VIX": 19.6, "ret": -0.004, ...}}

The stream is deliberately *volatile*, as real feeds are:
  * fields appear when their source starts (breakeven inflation 2003,
    dollar index 2006) - the schema grows over time;
  * outages: every field is dropped from a document with probability 2 %;
  * schema drift: from 2016 the provider renames ``vix`` to ``VIX`` and
    reports the 10y-3m term spread in basis points as ``term_bp``.

``normalise`` is the ingestion layer that maps every document version to
one canonical schema; ``window`` reads a date range back as a DataFrame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tinydb import Query, TinyDB
from tinydb.storages import MemoryStorage

GROUPS = {
    "vol": ["log_pk_d", "log_pk_w", "log_pk_m"],
    "vix": ["log_vix", "vix_chg_5", "vrp"],
    "mom": ["ret_5", "ret_22", "ret_66", "drawdown"],
    "rates": ["term_10y3m", "term_10y2y", "dgs10_chg_22", "ff_rate"],
    "credit": ["credit_baa_aaa", "credit_baa_10y"],
    "macro": ["oil_ret_22", "breakeven_10y", "usd_ret_22", "volume_z"],
}
FEATURES = [f for g in GROUPS.values() for f in g]
DRIFT_DATE = "2016-01-01"
OUTAGE_P = 0.02


def canonical_frame(feats: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=feats.index)
    for c in ("pk_d", "pk_w", "pk_m"):
        d[f"log_{c}"] = np.log(feats[c].clip(lower=1e-3))
    d["log_vix"] = np.log(feats["vix"])
    for c in ["vix_chg_5", "vrp", "ret_5", "ret_22", "ret_66", "drawdown", "term_10y3m", "term_10y2y",
              "dgs10_chg_22", "ff_rate", "credit_baa_aaa", "credit_baa_10y", "oil_ret_22",
              "breakeven_10y", "usd_ret_22", "volume_z"]:
        d[c] = feats[c]
    d["y"] = feats["y_logrv_5"]
    return d


def to_documents(frame: pd.DataFrame, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    docs = []
    for t, row in frame.iterrows():
        fields = {}
        for k, v in row.items():
            if k == "y" or pd.isna(v) or rng.uniform() < OUTAGE_P:
                continue
            fields[k] = float(v)
        if t >= pd.Timestamp(DRIFT_DATE):            # schema drift
            if "log_vix" in fields:
                fields["VIX"] = float(np.exp(fields.pop("log_vix")))
            if "term_10y3m" in fields:
                fields["term_bp"] = 100.0 * fields.pop("term_10y3m")
        doc = {"date": t.strftime("%Y-%m-%d"), "src": "feed-B" if t >= pd.Timestamp(DRIFT_DATE) else "feed-A",
               "fields": fields}
        if not pd.isna(row["y"]):
            doc["label"] = float(row["y"])           # becomes known 5 days later (handled by reader)
        docs.append(doc)
    return docs


def normalise(fields: dict) -> dict:
    """Map any document version onto the canonical schema."""
    f = dict(fields)
    if "VIX" in f:
        f["log_vix"] = float(np.log(f.pop("VIX")))
    if "term_bp" in f:
        f["term_10y3m"] = f.pop("term_bp") / 100.0
    return f


class MarketStore:
    def __init__(self, path: Path | None = None):
        self.db = TinyDB(path) if path else TinyDB(storage=MemoryStorage)
        self.t = self.db.table("days")

    def ingest(self, docs: list[dict]):
        self.t.truncate()
        self.t.insert_multiple(docs)

    def window(self, start: str, end: str) -> pd.DataFrame:
        """Documents with start <= date <= end, normalised, as a DataFrame."""
        q = Query()
        rows = self.t.search((q.date >= start) & (q.date <= end))
        recs = {r["date"]: {**normalise(r["fields"]), "y": r.get("label", np.nan)} for r in rows}
        df = pd.DataFrame.from_dict(recs, orient="index")
        df.index = pd.to_datetime(df.index)
        return df.sort_index().reindex(columns=FEATURES + ["y"])

    def schema_versions(self) -> pd.DataFrame:
        """Which field names occur in which year - a quick NoSQL schema audit."""
        out = {}
        for r in self.t.all():
            y = r["date"][:4]
            out.setdefault(y, set()).update(r["fields"].keys())
        names = sorted(set().union(*out.values()))
        return pd.DataFrame({y: [n in out[y] for n in names] for y in sorted(out)}, index=names)
