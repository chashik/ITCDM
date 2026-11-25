"""Rebuild paper/generated/*_numbers.tex from results/*_numbers.json (no recomputation)
and normalise table header rules.  Run:  python -m dmfin.reexport"""
import json
import re

from . import config as C
from .reporting import save_numbers, wrap_table_heads

for js in sorted(C.RESULTS_DIR.glob("*_numbers.json")):
    lab = js.name[: -len("_numbers.json")]
    save_numbers(lab, json.loads(js.read_text()))
for tex in C.GEN_DIR.glob("*.tex"):
    if tex.name.endswith("_numbers.tex"):          # macros: written by save_numbers only
        continue
    s = tex.read_text()
    s2 = re.sub(r"\\midrule(?!\\midrule)", r"\\hdrule", s) if "\\hdrule" not in s else s
    s2 = re.sub(r"(?<=[&\s{])-(?=\d)", "\u2212", s2)   # proper minus sign in tables
    s2 = wrap_table_heads(s2)                         # two-line heads: tables at 13 pt without scaling
    tex.write_text(s2)
print("re-exported")
