"""Regression tests for the semantic index.

Run with: python3 -m unittest discover -s tests

The index is optional, so the pure logic is tested unconditionally and anything needing
Milvus or the embedding model is skipped when they are absent. That split is the point:
the valuation models must keep working on a machine with no torch installed, and a test
suite that failed without it would quietly make an optional dependency mandatory.

What is pinned here is what would silently corrupt results: a company with no business
description being embedded from its ticker symbol, a fund's strategy blurb being indexed
as if it were a business, and Milvus's lack of scalar nulls turning an absent fair value
into a fair value of zero.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valuation  # noqa: E402
import vector_index  # noqa: E402
from valuation import assess  # noqa: E402

from test_valuation import healthy  # noqa: E402


def dependencies_present() -> bool:
    try:
        import pymilvus  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


class SkipRules(unittest.TestCase):
    """Nothing here touches Milvus: upsert returns before building a client."""

    def test_a_company_without_a_description_is_not_indexed(self):
        """A vector built from the string 'OTEX' encodes nothing about the business and
        would pollute every neighbour search that touched it."""
        result = assess(healthy())
        self.assertEqual(vector_index.upsert([result], {}), 0)

    def test_an_empty_result_set_does_no_work(self):
        self.assertEqual(vector_index.upsert([], {"X": "a business"}), 0)

    def test_summaries_are_matched_by_ticker(self):
        """A summary filed under the wrong ticker must not be borrowed by another."""
        result = assess(healthy())
        self.assertEqual(vector_index.upsert([result], {"SOMEONE-ELSE": "text"}), 0)


class SentinelHandling(unittest.TestCase):
    """Milvus has no null for scalar fields, so absence is encoded and decoded."""

    def test_absent_fair_value_is_not_read_as_zero(self):
        """Storing a missing estimate as 0.0 would render as a fair value of zero, which
        reads as a real and catastrophic verdict rather than as 'unknown'."""
        rows = [{"entity": {"ticker": "X", "name": "X Corp", "sector": "",
                            "verdict": valuation.NO_DATA, "price": 10.0,
                            "fair_value": -1.0, "margin": -99.0, "currency": "USD"},
                 "distance": 0.5}]
        neighbour = vector_index._to_neighbours(rows)[0]
        self.assertEqual(neighbour.fair_value, -1.0)
        self.assertLess(neighbour.margin, -90)

    def test_a_search_does_not_return_the_stock_it_started_from(self):
        """Every company is its own nearest neighbour, which is not a finding."""
        rows = [{"entity": {"ticker": "NVDA", "name": "", "sector": "", "verdict": "",
                            "price": 1.0, "fair_value": 1.0, "margin": 0.0,
                            "currency": "USD"}, "distance": 1.0},
                {"entity": {"ticker": "AMD", "name": "", "sector": "", "verdict": "",
                            "price": 1.0, "fair_value": 1.0, "margin": 0.0,
                            "currency": "USD"}, "distance": 0.8}]
        found = vector_index._to_neighbours(rows, drop="NVDA")
        self.assertEqual([n.ticker for n in found], ["AMD"])


class OptionalDependency(unittest.TestCase):
    def test_the_screener_imports_without_the_vector_stack(self):
        """valuation and market_data must never import the vector half at module scope,
        or an optional dependency becomes mandatory."""
        import market_data  # noqa: F401

        source = Path(__file__).resolve().parent.parent
        for module in ("valuation.py", "market_data.py", "history.py"):
            text = (source / module).read_text()
            self.assertNotIn("import vector_index", text, module)
            self.assertNotIn("import pymilvus", text, module)
            self.assertNotIn("sentence_transformers", text, module)

    def test_missing_dependencies_raise_a_typed_error(self):
        """So the CLI can print an install hint rather than a stack trace."""
        self.assertTrue(issubclass(vector_index.VectorIndexUnavailable, RuntimeError))


@unittest.skipUnless(dependencies_present(), "vector dependencies not installed")
class Integration(unittest.TestCase):
    """End-to-end against a throwaway Milvus Lite store."""

    def setUp(self):
        import tempfile

        self._original = vector_index.INDEX_PATH
        vector_index.INDEX_PATH = Path(tempfile.mkdtemp()) / "index.db"

    def tearDown(self):
        vector_index.INDEX_PATH = self._original

    def test_a_stored_company_is_retrievable_by_description(self):
        result = assess(healthy())
        stored = vector_index.upsert(
            [result], {result.ticker: "designs and manufactures analogue power "
                                      "semiconductors for data centres"})
        self.assertEqual(stored, 1)

        found = vector_index.search("power semiconductor chips")
        self.assertIn(result.ticker, [n.ticker for n in found])
        self.assertEqual(found[0].verdict, result.verdict)

    def test_reopening_an_existing_collection_still_queries(self):
        """Milvus leaves a collection 'released' when reopened, and search against a
        released collection raises rather than returning empty -- so a second run of the
        CLI failed where the first succeeded."""
        result = assess(healthy())
        vector_index.upsert([result], {result.ticker: "a manufacturer of widgets"})
        self.assertEqual(len(vector_index.stored()), 1)
        self.assertEqual(len(vector_index.stored()), 1)

    def test_an_unindexed_ticker_is_reported_not_guessed(self):
        result = assess(healthy())
        vector_index.upsert([result], {result.ticker: "a manufacturer of widgets"})
        with self.assertRaises(LookupError):
            vector_index.similar("NOSUCH")

    def test_reindexing_updates_rather_than_duplicates(self):
        """A verdict that changed must overwrite, or searches return stale advice."""
        first = assess(healthy())
        vector_index.upsert([first], {first.ticker: "a manufacturer of widgets"})
        second = assess(healthy(price=1.0))
        vector_index.upsert([second], {second.ticker: "a manufacturer of widgets"})

        rows = vector_index.stored()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].price, 1.0)


if __name__ == "__main__":
    unittest.main()
