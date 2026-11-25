"""Manifest of the downloaded data (``data/raw``), shared with the code instead of the data.

The data of the guide are not redistributed: ``make data`` downloads them from their providers.
Providers revise their data (adjusted prices, revised series, rewritten web pages), so a new
download may differ from the one on which the numbers of the guide were computed. The manifest
records, for every downloaded file, its size, number of lines after the header, first and last date and
SHA-256 checksum, and the date of the local copy; ``--check`` compares a new download with it.

    python -m dmfin.manifest            # write data_manifest.csv from data/raw
    python -m dmfin.manifest --check    # compare data/raw with data_manifest.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

from . import config as C

MANIFEST = Path(__file__).resolve().parent.parent / "data_manifest.csv"
PROVIDER = {"yahoo": "Yahoo Finance", "fred": "FRED (Federal Reserve Bank of St. Louis)",
            "fredmd": "FRED-MD (McCracken & Ng)", "french": "Kenneth R. French Data Library",
            "edgar": "SEC EDGAR", "fomc": "Federal Reserve Board (FOMC calendars)",
            "uci": "UCI Machine Learning Repository"}
D9 = {"belstat": "Belstat", "bis": "BIS", "cbr": "Bank of Russia", "ecb": "ECB", "ec_oil": "EU Oil Bulletin",
      "eec": "Eurasian Economic Commission", "eurostat": "Eurostat", "fao": "FAO FPMA", "gus": "Statistics Poland",
      "imf": "IMF", "manual": "manual download (lab10_economies/collect/fetch_blocked.sh)",
      "minfin_by": "Ministry of Finance of Belarus", "moex": "Moscow Exchange", "nbp": "Narodowy Bank Polski", "nbrb": "National Bank of Belarus",
      "nbrb_api": "National Bank of Belarus (API)", "obligacje": "obligacjeskarbowe.pl", "oecd": "OECD"}
TEXT = {".csv", ".data", ".txt"}
LOGS = {"fetch_log.txt"}                          # logs of the download scripts: not data, change on every run
DATE = re.compile(r"^(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})")


def _describe(path: Path) -> dict:
    raw = path.read_bytes()
    lines, first, last = "", "", ""
    if path.suffix.lower() in TEXT:
        rows = [r for r in raw.decode("utf-8", errors="replace").splitlines() if r.strip()]
        lines = len(rows) - (path.suffix.lower() != ".data")   # without the header line (.data files have none)
        dates = [m.group(1) for r in rows if (m := DATE.match(r.split(",")[0].strip('"')))]
        first, last = (dates[0], dates[-1]) if dates else ("", "")
    rel = path.relative_to(C.RAW_DIR)
    provider = ("D9: " + D9.get(rel.parts[1], rel.parts[1])) if rel.parts[0] == "d9" else PROVIDER.get(rel.parts[0], rel.parts[0])
    return {"file": rel.as_posix(), "provider": provider, "bytes": len(raw),
            "data_lines": lines, "first_date": first, "last_date": last,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "local_copy": dt.date.fromtimestamp(path.stat().st_mtime).isoformat()}


def build() -> pd.DataFrame:
    files = sorted(p for p in C.RAW_DIR.rglob("*") if p.is_file() and p.name not in LOGS)
    return pd.DataFrame([_describe(p) for p in files])


def write(path: Path = MANIFEST) -> Path:
    df = build()
    if df.empty:
        sys.exit(f"no data in {C.RAW_DIR}; run `make data` first")
    df.to_csv(path, index=False)
    return path


def check(path: Path = MANIFEST) -> int:
    """Print the differences between data/raw and the manifest; return the number of differing files."""
    ref = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("file")
    new = build().astype(str).set_index("file") if any(C.RAW_DIR.rglob("*")) else ref.iloc[:0]
    missing = ref.index.difference(new.index)
    extra = new.index.difference(ref.index)
    common = ref.index.intersection(new.index)
    changed = [f for f in common if ref.at[f, "sha256"] != new.at[f, "sha256"]]
    for f in missing:
        print(f"missing   {f}")
    for f in extra:
        print(f"new       {f}")
    for f in changed:
        a, b = ref.loc[f], new.loc[f]
        detail = (f"lines {a.data_lines} -> {b.data_lines}, dates {a.first_date}..{a.last_date} -> "
                  f"{b.first_date}..{b.last_date}" if a.data_lines else "content differs")
        print(f"changed   {f}: {detail}")
    print(f"{len(common) - len(changed)} of {len(ref)} files identical to the data of the guide; "
          f"{len(changed)} changed, {len(missing)} missing, {len(extra)} not in the manifest")
    if changed or len(missing):
        print("Results computed on these data may differ from the numbers printed in the guide.")
    return len(changed) + len(missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare data/raw with the manifest")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if check() else 0)
    print(write())
