"""Lab 9 tests: ontology, OWL 2 RL inferences, SHACL validation, SPARQL."""
import pandas as pd
from rdflib import RDF, Literal

from lab9_kg.kg import DM, add_company, add_result, ontology, reason, validate

ROW = {"ticker": "ABC", "name": "ABC Corp", "cik": 123, "sector": "Energy", "sic": 2911,
       "sic_description": "Petroleum Refining", "ff49": "Oil", "exchanges": "NYSE", "risk_cluster": "C1",
       "comove_cluster": "F2"}


def graph(**over):
    g = ontology()
    add_company(g, pd.Series({**ROW, **over}))
    return g


def test_property_chain_and_subproperty_and_domain_are_inferred():
    g = graph()
    h = reason(g)
    c = DM["company/ABC"]
    assert (c, DM.sicDivisionOf, DM["sicdivision/D"]) in h          # hasSIC o inDivision (2911 -> manufacturing)
    assert (c, DM.memberOf, DM["cluster/C1"]) in h                    # inRiskCluster is a sub-property of memberOf
    assert (DM["sic/2911"], RDF.type, DM.SICIndustry) in h            # range of hasSIC


def test_symmetric_property():
    g = graph()
    add_company(g, pd.Series({**ROW, "ticker": "XYZ", "cik": 456}))
    g.add((DM["company/ABC"], DM.coMovesWith, DM["company/XYZ"]))
    assert (DM["company/XYZ"], DM.coMovesWith, DM["company/ABC"]) in reason(g)


def test_shacl_accepts_a_complete_company_and_rejects_a_missing_cik():
    assert validate(graph())[0]
    g = graph()
    g.remove((DM["company/ABC"], DM.cik, None))
    conforms, _, text = validate(g)
    assert not conforms and "cik" in text


def test_sparql_selects_results_by_metric():
    g = ontology()
    add_result(g, 6, "HAR", "test RMSE", 0.415)
    add_result(g, 6, "LSTM", "test RMSE", 0.406)
    q = """PREFIX dm: <http://example.org/gsu-dm#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?m WHERE { ?r dm:value ?v ; dm:ofModel/rdfs:label ?m } ORDER BY ?v LIMIT 1"""
    assert [str(r[0]) for r in g.query(q)] == ["LSTM"]
