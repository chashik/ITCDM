"""Lab 9. A knowledge graph of the research domain (RDF, RDFS/OWL, SHACL, SPARQL).

``ontology()`` returns the schema: classes, properties, domains and ranges, a class
hierarchy, a symmetric property and a property chain.  ``build()`` adds the
instances: companies with their SEC metadata and classifications, data series
with their sources, clusters of Lab 1 and results of the laboratory assignments.
``reason()`` computes the OWL 2 RL closure; ``validate()`` checks SHACL shapes.
"""
from __future__ import annotations

import re

import pandas as pd
from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL

DM = Namespace("http://example.org/gsu-dm#")

ONTOLOGY_TTL = """
@prefix dm:   <http://example.org/gsu-dm#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

dm:Organization a owl:Class .
dm:Company a owl:Class ; rdfs:subClassOf dm:Organization ; rdfs:label "listed company" .
dm:DataProvider a owl:Class ; rdfs:subClassOf dm:Organization .
dm:Classification a owl:Class .
dm:Sector a owl:Class ; rdfs:subClassOf dm:Classification ; rdfs:label "GICS sector" .
dm:SICIndustry a owl:Class ; rdfs:subClassOf dm:Classification ; rdfs:label "SIC industry (4 digits)" .
dm:SICDivision a owl:Class ; rdfs:subClassOf dm:Classification ; rdfs:label "SIC division" .
dm:FFIndustry a owl:Class ; rdfs:subClassOf dm:Classification ; rdfs:label "Fama-French industry (49)" .
dm:Exchange a owl:Class .
dm:Series a owl:Class ; rdfs:label "data series" .
dm:PriceSeries a owl:Class ; rdfs:subClassOf dm:Series .
dm:MacroSeries a owl:Class ; rdfs:subClassOf dm:Series .
dm:VolatilityIndex a owl:Class ; rdfs:subClassOf dm:Series .
dm:Cluster a owl:Class .
dm:RiskCluster a owl:Class ; rdfs:subClassOf dm:Cluster .
dm:ComovementCluster a owl:Class ; rdfs:subClassOf dm:Cluster .
dm:Model a owl:Class .
dm:Result a owl:Class .
dm:LabAssignment a owl:Class .

dm:inSector a owl:ObjectProperty ; rdfs:domain dm:Company ; rdfs:range dm:Sector .
dm:hasSIC a owl:ObjectProperty ; rdfs:domain dm:Company ; rdfs:range dm:SICIndustry .
dm:inDivision a owl:ObjectProperty ; rdfs:domain dm:SICIndustry ; rdfs:range dm:SICDivision .
dm:sicDivisionOf a owl:ObjectProperty ; owl:propertyChainAxiom ( dm:hasSIC dm:inDivision ) .
dm:inFFIndustry a owl:ObjectProperty ; rdfs:domain dm:Company ; rdfs:range dm:FFIndustry .
dm:listedOn a owl:ObjectProperty ; rdfs:domain dm:Company ; rdfs:range dm:Exchange .
dm:memberOf a owl:ObjectProperty ; rdfs:range dm:Cluster .
dm:inRiskCluster a owl:ObjectProperty ; rdfs:subPropertyOf dm:memberOf ; rdfs:range dm:RiskCluster .
dm:inComovementCluster a owl:ObjectProperty ; rdfs:subPropertyOf dm:memberOf ; rdfs:range dm:ComovementCluster .
dm:coMovesWith a owl:ObjectProperty , owl:SymmetricProperty ; rdfs:domain dm:Company ; rdfs:range dm:Company .
dm:providedBy a owl:ObjectProperty ; rdfs:domain dm:Series ; rdfs:range dm:DataProvider .
dm:usedIn a owl:ObjectProperty ; rdfs:domain dm:Series ; rdfs:range dm:LabAssignment .
dm:ofModel a owl:ObjectProperty ; rdfs:domain dm:Result ; rdfs:range dm:Model .
dm:inLab a owl:ObjectProperty ; rdfs:domain dm:Result ; rdfs:range dm:LabAssignment .

dm:cik a owl:DatatypeProperty ; rdfs:domain dm:Company ; rdfs:range xsd:integer .
dm:ticker a owl:DatatypeProperty ; rdfs:range xsd:string .
dm:sicCode a owl:DatatypeProperty ; rdfs:domain dm:SICIndustry ; rdfs:range xsd:integer .
dm:startDate a owl:DatatypeProperty ; rdfs:domain dm:Series ; rdfs:range xsd:date .
dm:frequency a owl:DatatypeProperty ; rdfs:domain dm:Series .
dm:metric a owl:DatatypeProperty ; rdfs:domain dm:Result .
dm:value a owl:DatatypeProperty ; rdfs:domain dm:Result ; rdfs:range xsd:decimal .
"""

# SIC divisions (first two digits of the SIC code)
SIC_DIVISIONS = [(1, 9, "A", "Agriculture, forestry and fishing"), (10, 14, "B", "Mining"), (15, 17, "C", "Construction"),
                 (20, 39, "D", "Manufacturing"), (40, 49, "E", "Transportation, communications and utilities"),
                 (50, 51, "F", "Wholesale trade"), (52, 59, "G", "Retail trade"),
                 (60, 67, "H", "Finance, insurance and real estate"), (70, 89, "I", "Services"),
                 (91, 99, "J", "Public administration")]

SHAPES_TTL = """
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix dm:  <http://example.org/gsu-dm#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

dm:CompanyShape a sh:NodeShape ;
    sh:targetClass dm:Company ;
    sh:property [ sh:path dm:cik ; sh:datatype xsd:integer ; sh:minCount 1 ; sh:maxCount 1 ] ;
    sh:property [ sh:path dm:ticker ; sh:minCount 1 ; sh:pattern "^[A-Z.]{1,6}$" ] ;
    sh:property [ sh:path dm:inSector ; sh:class dm:Sector ; sh:minCount 1 ; sh:maxCount 1 ] ;
    sh:property [ sh:path dm:hasSIC ; sh:minCount 1 ; sh:maxCount 1 ] ;
    sh:property [ sh:path dm:inFFIndustry ; sh:minCount 1 ; sh:maxCount 1 ] .

dm:SeriesShape a sh:NodeShape ;
    sh:targetClass dm:Series ;
    sh:property [ sh:path dm:providedBy ; sh:minCount 1 ] ;
    sh:property [ sh:path dm:startDate ; sh:datatype xsd:date ; sh:minCount 1 ; sh:maxCount 1 ] ;
    sh:property [ sh:path dm:frequency ; sh:in ( "daily" "weekly" "monthly" ) ] .

dm:ResultShape a sh:NodeShape ;
    sh:targetClass dm:Result ;
    sh:property [ sh:path dm:value ; sh:minCount 1 ; sh:maxCount 1 ] ;
    sh:property [ sh:path dm:ofModel ; sh:minCount 1 ] .
"""


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def ontology() -> Graph:
    g = Graph()
    g.parse(data=ONTOLOGY_TTL, format="turtle")
    g.bind("dm", DM)
    return g


def add_company(g: Graph, row: pd.Series):
    c = DM[f"company/{slug(row['ticker'])}"]
    g.add((c, RDF.type, DM.Company))
    g.add((c, RDFS.label, Literal(row["name"])))
    g.add((c, DM.ticker, Literal(row["ticker"])))
    if pd.notna(row.get("cik")):
        g.add((c, DM.cik, Literal(int(row["cik"]), datatype=XSD.integer)))
    if pd.notna(row.get("sector")):
        s = DM[f"sector/{slug(row['sector'])}"]
        g.add((s, RDF.type, DM.Sector)); g.add((s, RDFS.label, Literal(row["sector"])))
        g.add((c, DM.inSector, s))
    if pd.notna(row.get("sic")):
        code = int(row["sic"])
        i = DM[f"sic/{code}"]
        g.add((i, DM.sicCode, Literal(code, datatype=XSD.integer)))
        g.add((i, RDFS.label, Literal(row.get("sic_description") or str(code))))
        g.add((c, DM.hasSIC, i))
        for lo, hi, letter, name in SIC_DIVISIONS:
            if lo <= code // 100 <= hi:
                d = DM[f"sicdivision/{letter}"]
                g.add((d, RDFS.label, Literal(name)))
                g.add((i, DM.inDivision, d))
    if pd.notna(row.get("ff49")):
        f = DM[f"ff49/{slug(row['ff49'])}"]
        g.add((f, RDF.type, DM.FFIndustry)); g.add((f, RDFS.label, Literal(row["ff49"])))
        g.add((c, DM.inFFIndustry, f))
    for ex in str(row.get("exchanges") or "").split(";"):
        if ex and ex != "nan":
            e = DM[f"exchange/{slug(ex)}"]
            g.add((e, RDF.type, DM.Exchange)); g.add((e, RDFS.label, Literal(ex)))
            g.add((c, DM.listedOn, e))
    for prop, col, cls in ((DM.inRiskCluster, "risk_cluster", None), (DM.inComovementCluster, "comove_cluster", None)):
        if pd.notna(row.get(col)):
            k = DM[f"cluster/{row[col]}"]
            g.add((k, RDFS.label, Literal(row[col])))
            g.add((c, prop, k))
    return c


def add_series(g: Graph, sid: str, label: str, cls, provider: str, start, frequency: str, labs=()):
    s = DM[f"series/{slug(sid)}"]
    g.add((s, RDF.type, cls)); g.add((s, RDFS.label, Literal(label)))
    p = DM[f"provider/{slug(provider)}"]
    g.add((p, RDF.type, DM.DataProvider)); g.add((p, RDFS.label, Literal(provider)))
    g.add((s, DM.providedBy, p))
    if start is not None and pd.notna(start):
        g.add((s, DM.startDate, Literal(pd.Timestamp(start).date(), datatype=XSD.date)))
    g.add((s, DM.frequency, Literal(frequency)))
    for lab in labs:
        lw = DM[f"lab/{lab}"]
        g.add((lw, RDF.type, DM.LabAssignment)); g.add((lw, RDFS.label, Literal(f"Lab {lab}")))
        g.add((s, DM.usedIn, lw))
    return s


def add_result(g: Graph, lab: int, model: str, metric: str, value: float):
    r = DM[f"result/lab{lab}_{slug(model)}_{slug(metric)}"]
    m = DM[f"model/{slug(model)}"]
    lw = DM[f"lab/{lab}"]
    g.add((r, RDF.type, DM.Result)); g.add((m, RDF.type, DM.Model)); g.add((m, RDFS.label, Literal(model)))
    g.add((lw, RDF.type, DM.LabAssignment)); g.add((lw, RDFS.label, Literal(f"Lab {lab}")))
    g.add((r, DM.ofModel, m)); g.add((r, DM.inLab, lw)); g.add((r, DM.metric, Literal(metric)))
    g.add((r, DM.value, Literal(round(float(value), 6), datatype=XSD.decimal)))
    return r


def reason(g: Graph) -> Graph:
    """OWL 2 RL closure (rule-based materialization) of a copy of the graph."""
    import owlrl
    h = Graph()
    for t in g:
        h.add(t)
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(h)
    return h


def validate(g: Graph):
    from pyshacl import validate as shacl
    shapes = Graph().parse(data=SHAPES_TTL, format="turtle")
    conforms, report, text = shacl(g, shacl_graph=shapes, inference="rdfs")
    return conforms, report, text
