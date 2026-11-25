"""Lab 9. Knowledge graph of the domain: construction, reasoning, validation, queries, graph mining.

Run:  python -m lab9_kg.run
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from rdflib import RDF, Literal
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from dmfin import config as C
from dmfin import plotting as P
from dmfin.config import FRED_SERIES
from dmfin.data import load_stock_prices
from dmfin.data_ext import MULTI_ASSET, VOL_FRED, VOL_YAHOO, inventory
from dmfin.reporting import save_numbers, save_table

from .kg import DM, add_company, add_result, add_series, ontology, reason, validate

Q = Path(__file__).resolve().parent / "queries"
QUERIES = {
    "q1_sector_division": """PREFIX dm: <http://example.org/gsu-dm#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?sector ?division (COUNT(?c) AS ?n) WHERE {
  ?c a dm:Company ; dm:inSector/rdfs:label ?sector ; dm:sicDivisionOf/rdfs:label ?division .
} GROUP BY ?sector ?division ORDER BY ?sector DESC(?n)""",
    "q2_outliers": """PREFIX dm: <http://example.org/gsu-dm#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?ticker ?sector ?industry WHERE {
  ?c dm:ticker ?ticker ; dm:inSector ?s ; dm:inFFIndustry ?f .
  ?s rdfs:label ?sector . ?f rdfs:label ?industry .
  { SELECT ?s (COUNT(?o) AS ?total) WHERE { ?o dm:inSector ?s } GROUP BY ?s }
  { SELECT ?s ?f (COUNT(?o2) AS ?k) WHERE { ?o2 dm:inSector ?s ; dm:inFFIndustry ?f } GROUP BY ?s ?f }
  FILTER (?k = 1 && ?total >= 5)
} ORDER BY ?sector ?ticker""",
    "q3_late_series": """PREFIX dm: <http://example.org/gsu-dm#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?series ?provider ?start WHERE {
  ?s a dm:Series ; rdfs:label ?series ; dm:startDate ?start ; dm:providedBy/rdfs:label ?provider .
  FILTER (?start > "2005-01-01"^^xsd:date)
} ORDER BY ?start""",
    "q4_best_models": """PREFIX dm: <http://example.org/gsu-dm#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?lab ?metric ?model ?value WHERE {
  ?r dm:inLab ?l ; dm:metric ?metric ; dm:value ?value ; dm:ofModel/rdfs:label ?model .
  ?l rdfs:label ?lab .
  { SELECT ?l ?metric (MIN(?v) AS ?best) WHERE { ?x dm:inLab ?l ; dm:metric ?metric ; dm:value ?v .
      FILTER (CONTAINS(?metric, "RMSE")) } GROUP BY ?l ?metric }
  FILTER (?value = ?best)
} ORDER BY ?lab""",
    "q5_two_hops": """PREFIX dm: <http://example.org/gsu-dm#>
SELECT DISTINCT ?ticker WHERE {
  ?n dm:ticker "NVDA" .
  ?n dm:coMovesWith/dm:coMovesWith? ?m .
  ?m dm:ticker ?ticker . FILTER (?m != ?n)
} ORDER BY ?ticker""",
    "q6_cluster_industries": """PREFIX dm: <http://example.org/gsu-dm#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?cluster (COUNT(DISTINCT ?f) AS ?industries) (COUNT(DISTINCT ?s) AS ?sectors) (COUNT(?c) AS ?companies) WHERE {
  ?c dm:inRiskCluster/rdfs:label ?cluster ; dm:inFFIndustry ?f ; dm:inSector ?s .
} GROUP BY ?cluster ORDER BY ?cluster""",
}


def sparql_table(g, q):
    res = g.query(q)
    return pd.DataFrame([[str(v) if v is not None else "" for v in row] for row in res],
                        columns=[str(v) for v in res.vars])


def main():
    num = {}
    Q.mkdir(exist_ok=True)
    for k, q in QUERIES.items():
        (Q / f"{k}.rq").write_text(q + "\n")

    # ---------------- construction
    g = ontology()
    n_onto = len(g)
    meta = pd.read_csv(C.RESULTS_DIR / "lab0_edgar_meta.csv")
    cl = pd.read_csv(C.RESULTS_DIR / "lab1_clusters.csv")
    comp = meta.merge(cl, on="ticker", how="left")
    for _, row in comp.iterrows():
        add_company(g, row)
    px = load_stock_prices().loc[C.STOCKS_START:C.STOCKS_SPLIT]
    rho = np.log(px).diff().corr()
    edges = set()
    for t in rho.index:                                        # five strongest co-movements of each stock
        for u in rho[t].drop(t).nlargest(5).index:
            edges.add(tuple(sorted((t, u))))
    for a, b in edges:
        g.add((DM[f"company/{a}"], DM.coMovesWith, DM[f"company/{b}"]))
    inv = inventory()
    for t, name in MULTI_ASSET.items():
        first = inv.loc[inv["source"] == name, "first"]
        add_series(g, t, name, DM.PriceSeries, "Yahoo Finance", first.iloc[0] if len(first) else None, "daily", labs=(0, 6, 7))
    for k, name in {**VOL_YAHOO, **VOL_FRED}.items():
        first = inv.loc[inv["source"] == name, "first"]
        freq = "weekly" if k in ("STLFSI4", "NFCI") else "daily"
        add_series(g, k, name, DM.VolatilityIndex if "vol" in name.lower() or "VIX" in name else DM.MacroSeries,
                   "Yahoo Finance" if k in VOL_YAHOO else "FRED", first.iloc[0] if len(first) else None, freq, labs=(0, 7))
    for k, name in FRED_SERIES.items():
        add_series(g, k, name, DM.MacroSeries, "FRED", pd.Timestamp("2000-01-03") if k not in ("T10YIE", "DTWEXBGS")
                   else pd.Timestamp({"T10YIE": "2003-01-02", "DTWEXBGS": "2006-01-02"}[k]),
                   "monthly" if k == "USREC" else "daily", labs=(0, 2, 6, 7))
    add_series(g, "FF49", "49 industry portfolios (value-weighted)", DM.PriceSeries, "Kenneth R. French Data Library",
               pd.Timestamp("2000-01-03"), "daily", labs=(1, 4, 6))
    add_series(g, "FREDMD", "FRED-MD (126 monthly series)", DM.MacroSeries, "FRED-MD", pd.Timestamp("2000-01-01"),
               "monthly", labs=(2,))
    for lab, f, col, metric in ((2, "lab2_compare.csv", "Test RMSE", "test RMSE of log RV (22 days)"),
                                (4, "lab4_vol_trees.csv", "Test RMSE", "test RMSE of log RV (22 days)"),
                                (6, "lab6_forecast.csv", "Test RMSE", "test RMSE of log RV (5 days)"),
                                (3, "lab3_ext_results.csv", "AUC", "test AUC (credit default)"),
                                (4, "lab4_compare.csv", "AUC", "test AUC (credit default)"),
                                (4, "lab4_global.csv", "Mean test RMSE", "mean test RMSE per series of log RV (5 days, 62 series)"),
                                (6, "lab6_global.csv", "Test RMSE", "pooled test RMSE of log RV (5 days, 62 series)"),
                                (7, "lab7x_results.csv", "RMSE, 11", "test RMSE of log RV (5 days), 11 feature groups"),
                                (8, "lab8x_search.csv", "Test", "pooled test RMSE of log RV (5 days, 62 series)")):
        p = C.RESULTS_DIR / f
        if p.exists():
            t = pd.read_csv(p)
            for _, r in t.iterrows():
                v = pd.to_numeric(str(r[col]).replace("\u2212", "-"), errors="coerce")
                if "Series" in t.columns and str(r["Series"]) != "62":            # evaluated on one series only
                    continue
                if pd.notna(v) and "oracle" not in str(r.iloc[0]).lower():          # bounds are not models
                    add_result(g, lab, str(r.iloc[0]).replace("$", ""), metric, float(v))
    n_asserted = len(g)
    g.serialize(C.RESULTS_DIR / "lab9_kg.ttl", format="turtle")

    # ---------------- reasoning (OWL 2 RL closure)
    h = reason(g)
    inferred = len(h) - n_asserted
    n_chain = len(list(h.triples((None, DM.sicDivisionOf, None))))
    n_member = len(list(h.triples((None, DM.memberOf, None))))
    n_sym = len(list(h.triples((None, DM.coMovesWith, None))))
    n_series = len(set(h.subjects(RDF.type, DM.Series)))
    num.update({"triples_onto": n_onto, "triples_asserted": n_asserted, "triples_closure": len(h), "triples_inferred": inferred,
                "n_companies": len(comp), "n_edges": len(edges), "n_chain": n_chain, "n_member": n_member,
                "n_sym": n_sym, "n_series_inferred": n_series,
                "n_results": len(set(g.subjects(RDF.type, DM.Result)))})

    # ---------------- SHACL validation: the graph and a deliberately damaged copy
    conforms, _, _ = validate(g)
    bad = ontology()
    for t in g:
        bad.add(t)
    aapl = DM["company/AAPL"]
    bad.remove((aapl, DM.cik, None))
    bad.add((DM["company/MSFT"], DM.ticker, Literal("msft")))
    bad.add((DM["company/JPM"], DM.inSector, DM["sector/Financials_2"]))
    bad.remove((DM["series/FREDMD"], DM.frequency, None)); bad.add((DM["series/FREDMD"], DM.frequency, Literal("quarterly")))
    conf_bad, rep, txt = validate(bad)
    viol = []
    q = """PREFIX sh: <http://www.w3.org/ns/shacl#>
SELECT ?focus ?path ?comp WHERE { ?r a sh:ValidationResult ; sh:focusNode ?focus ; sh:resultPath ?path ;
  sh:sourceConstraintComponent ?comp . }"""
    for focus, path, constraint in rep.query(q):
        viol.append((str(focus).split("#")[-1], str(path).split("#")[-1],
                     str(constraint).split("#")[-1].replace("ConstraintComponent", "")))
    vt = pd.DataFrame(sorted(viol), columns=["Node", "Property", "Violated constraint"]).set_index("Node")
    save_table("lab9_shacl", vt, col_format="lll")
    num.update({"shacl_conforms": "yes" if conforms else "no", "shacl_bad_violations": len(viol)})

    # ---------------- SPARQL analytics
    res = {k: sparql_table(h, q) for k, q in QUERIES.items()}
    q1 = res["q1_sector_division"]
    piv = q1.pivot_table(index="sector", columns="division", values="n", aggfunc="first").fillna(0).astype(int)
    q2 = res["q2_outliers"]
    save_table("lab9_outliers", q2.set_index("ticker"), col_format="lll")
    q3 = res["q3_late_series"]
    q3["start"] = q3["start"].str[:10]
    save_table("lab9_late_series", q3.set_index("series"), col_format="p{8cm}ll")
    q4 = res["q4_best_models"]
    q4["value"] = q4["value"].astype(float).round(3)
    save_table("lab9_best_models", q4.set_index("lab"), nd={"value": 3}, col_format="lp{5.6cm}p{5.0cm}c")
    q5 = res["q5_two_hops"]
    q6 = res["q6_cluster_industries"]
    save_table("lab9_clusters", q6.set_index("cluster"), col_format="lccc")
    num.update({"q2_n": len(q2), "q3_n": len(q3), "q5_n": len(q5), "q5_list": ", ".join(q5["ticker"].head(12)),
                "q1_divisions": int(piv.shape[1])})
    f, ax = P.fig(3.4)
    im = ax.imshow(piv.to_numpy(), cmap=P.SEQ_CELL, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([textwrap.fill(c, 18) for c in piv.columns], fontsize=9)       # full names, wrapped
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if piv.iloc[i, j]:
                ax.text(j, i, int(piv.iloc[i, j]), ha="center", va="center", fontsize=9, color=P.INK)
    f.tight_layout(); P.save(f, "lab9_sector_division")

    # ---------------- graph mining: communities of the co-movement graph vs classifications
    G = nx.Graph()
    for a, b in sorted(edges):           # a fixed order: Louvain and the layout depend on the order of the edges
        G.add_edge(a, b, weight=float(rho.loc[a, b]))
    comm = nx.community.louvain_communities(G, weight="weight", seed=C.SEED)
    lab_comm = {n: i for i, c in enumerate(comm) for n in c}
    nodes = sorted(G.nodes)
    gics = comp.set_index("ticker")["sector"].reindex(nodes)
    ff = comp.set_index("ticker")["ff49"].reindex(nodes)
    lc = [lab_comm[n] for n in nodes]
    within = lambda lab: float(np.mean([rho.loc[a, b] for a in nodes for b in nodes if a < b and lab[a] == lab[b]]))
    num.update({"n_comm": len(comm), "modularity": nx.community.modularity(G, comm, weight="weight"),
                "nmi_gics": normalized_mutual_info_score(gics, lc), "nmi_ff49": normalized_mutual_info_score(ff, lc),
                "ari_gics": adjusted_rand_score(gics, lc), "ari_ff49": adjusted_rand_score(ff, lc),
                "n_ff49": int(ff.nunique()), "within_gics": within(gics.to_dict()), "within_ff49": within(ff.to_dict()),
                "mean_corr": float(rho.to_numpy()[np.triu_indices(len(rho), 1)].mean())})
    bc = nx.betweenness_centrality(G, weight=None)
    top = sorted(bc, key=bc.get, reverse=True)[:5]
    num["hubs"] = ", ".join(top)
    pos = nx.spring_layout(G, weight="weight", seed=C.SEED, k=0.35)
    secs = sorted(gics.unique())
    col = {s: (P.CAT + [P.INK2, P.MUTED, "#8c564b"])[i] for i, s in enumerate(secs)}
    f, ax = P.fig(4.6)
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.3, alpha=0.4, edge_color=P.MUTED)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[40 + 600 * bc[n] for n in G.nodes],
                           node_color=[col[gics[n]] for n in G.nodes], linewidths=0.3, edgecolors="white")
    nx.draw_networkx_labels(G, pos, labels={n: n for n in top}, font_size=7, ax=ax)
    for s in secs:
        ax.scatter([], [], color=col[s], label=s, s=20)
    ax.legend(fontsize=9, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False); ax.axis("off")
    f.tight_layout(); P.save(f, "lab9_graph")
    save_numbers("lab9", num)
    print(num)


if __name__ == "__main__":
    main()
