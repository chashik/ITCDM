"""Lab 5 (iteration 2) tests: Apriori and association rules."""
import pandas as pd
import pytest

from lab5_fuzzy.apriori import apriori, rules

# the transactional data of Han, Kamber and Pei (Table 6.1): nine transactions, items I1..I5
T = [["I1", "I2", "I5"], ["I2", "I4"], ["I2", "I3"], ["I1", "I2", "I4"], ["I1", "I3"], ["I2", "I3"], ["I1", "I3"],
     ["I1", "I2", "I3", "I5"], ["I1", "I2", "I3"]]


def matrix():
    items = ["I1", "I2", "I3", "I4", "I5"]
    return pd.DataFrame([[i in t for i in items] for t in T], columns=items)


def test_apriori_finds_the_frequent_itemsets_of_the_textbook_example():
    freq, stats = apriori(matrix(), min_support=2 / 9, max_len=4)
    sets = {frozenset(s) for s in freq}
    pairs = {frozenset(p) for p in (("I1", "I2"), ("I1", "I3"), ("I1", "I5"), ("I2", "I3"), ("I2", "I4"), ("I2", "I5"))}
    triples = {frozenset(("I1", "I2", "I3")), frozenset(("I1", "I2", "I5"))}
    assert {s for s in sets if len(s) == 2} == pairs and {s for s in sets if len(s) == 3} == triples
    assert not any(len(s) == 4 for s in sets)
    assert stats.loc[stats["size"] == 3, "candidates"].item() == 2      # the Apriori property prunes the other triples


def test_rule_measures():
    freq, _ = apriori(matrix(), min_support=2 / 9, max_len=3)
    r = rules(freq, ["I5"])
    row = r[r["antecedent"] == "I1 AND I2"].iloc[0]
    assert row["support"] == pytest.approx(2 / 9)
    assert row["confidence"] == pytest.approx(0.5)                     # 2 of the 4 transactions with I1 and I2
    assert row["lift"] == pytest.approx(0.5 / (2 / 9))
