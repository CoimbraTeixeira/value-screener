"""Regression tests for portfolio import and the holder's-view verdicts.

Run with: python3 -m unittest discover -s tests

Two classes of failure are covered. The importer must not read the wrong column when
Yahoo reshuffles its export -- an index-based parse would take 'Purchase Price' from
wherever it happened to sit and mis-state every cost basis silently. And the holder's
verdicts must not be a rename of the buy-side ones: a position up 121% in a business
that has started losing money is an EXIT, and the gain is exactly what makes that hard
to see.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import portfolio  # noqa: E402
import valuation  # noqa: E402
from portfolio import Holding, load_csv, review  # noqa: E402
from valuation import Fundamentals, assess  # noqa: E402

from test_valuation import agreeing, healthy  # noqa: E402


def write_csv(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return Path(handle.name)


class CsvImport(unittest.TestCase):
    def test_reads_a_yahoo_portfolio_export(self):
        path = write_csv(
            "Symbol,Current Price,Trade Date,Purchase Price,Quantity,Commission\n"
            "T,25.40,20240115,18.20,500,0.00\n"
            "KO,88.25,20200310,45.10,300,0.00\n")
        holdings = load_csv(path)
        self.assertEqual([h.ticker for h in holdings], ["T", "KO"])
        self.assertEqual(holdings[0].quantity, 500)
        self.assertEqual(holdings[0].cost_basis, 18.20)
        self.assertAlmostEqual(holdings[0].book_cost(), 9100.0)

    def test_columns_are_matched_by_name_not_position(self):
        """Same data, columns reordered and recapitalised. An index-based parse would
        read the price as the quantity here and never complain."""
        path = write_csv("quantity,SYMBOL,Purchase Price\n500,T,18.20\n")
        holding = load_csv(path)[0]
        self.assertEqual(holding.ticker, "T")
        self.assertEqual(holding.quantity, 500)
        self.assertEqual(holding.cost_basis, 18.20)

    def test_strips_totals_and_blank_rows(self):
        path = write_csv("Symbol,Quantity\nT,500\n,\nTOTAL,\nCASH,\n")
        self.assertEqual([h.ticker for h in load_csv(path)], ["T"])

    def test_parses_yahoo_number_formatting(self):
        """Exports carry thousands separators, currency marks and N/A."""
        path = write_csv('Symbol,Quantity,Purchase Price\nT,"1,500",$18.20\nKO,N/A,N/A\n')
        first, second = load_csv(path)
        self.assertEqual(first.quantity, 1500.0)
        self.assertEqual(first.cost_basis, 18.20)
        self.assertIsNone(second.quantity)

    def test_a_watchlist_export_with_symbols_only_still_loads(self):
        """Yahoo watchlists carry no quantity; those are screening inputs, not positions."""
        holdings = load_csv(write_csv("Symbol\nT\nKO\n"))
        self.assertEqual(len(holdings), 2)
        self.assertFalse(any(h.is_position for h in holdings))

    def test_a_file_without_a_symbol_column_is_an_error(self):
        """Better to fail than to screen an empty list and report 'nothing to buy'."""
        with self.assertRaises(ValueError):
            load_csv(write_csv("Name,Price\nAT&T,25.40\n"))


class HoldingShape(unittest.TestCase):
    def test_zero_quantity_is_a_watch_not_a_position(self):
        self.assertFalse(Holding("T", quantity=0).is_position)
        self.assertTrue(Holding("T", quantity=1).is_position)

    def test_book_cost_needs_both_halves(self):
        self.assertIsNone(Holding("T", quantity=100).book_cost())
        self.assertIsNone(Holding("T", cost_basis=10.0).book_cost())
        self.assertEqual(Holding("T", quantity=100, cost_basis=10.0).book_cost(), 1000.0)


class HolderVerdicts(unittest.TestCase):
    def test_a_broken_business_exits_however_large_the_gain(self):
        """Gilead screened AVOID while a holder sat on a 121% gain. The gain is not a
        reason to keep owning a business that has stopped earning."""
        broken = assess(healthy(price=150.0, eps_trailing=-2.0, free_cash_flow=-1.0e9))
        self.assertEqual(broken.verdict, valuation.AVOID)
        result = review(broken, Holding("GILD", quantity=150, cost_basis=68.0))
        self.assertEqual(result.action, portfolio.EXIT)
        self.assertGreater(result.unrealised_pct, 1.0)

    def test_a_large_premium_to_fair_value_trims(self):
        rich = assess(healthy(price=100000.0))
        result = review(rich, Holding("X", quantity=10, cost_basis=1.0))
        self.assertEqual(result.action, portfolio.TRIM)
        self.assertIn("above estimated fair value", result.rationale)

    def test_a_still_cheap_holding_says_add(self):
        result = review(assess(agreeing()), Holding("T", quantity=500, cost_basis=18.2))
        self.assertEqual(result.action, portfolio.ADD)

    def test_selling_needs_a_wider_margin_than_not_buying(self):
        """A stock slightly above fair value is HOLD, not TRIM: declining to buy is free,
        selling costs spread and tax."""
        base = assess(healthy())
        just_over = base.fair_value * (1 + portfolio.TRIM_PREMIUM / 2)
        nudged = assess(healthy(price=just_over))
        self.assertEqual(review(nudged, Holding("X", quantity=1, cost_basis=1.0)).action,
                         portfolio.HOLD)

    def test_trim_threshold_is_configurable(self):
        base = assess(healthy())
        priced = assess(healthy(price=base.fair_value * 1.3))
        holding = Holding("X", quantity=1, cost_basis=1.0)
        self.assertEqual(review(priced, holding, trim_premium=0.25).action, portfolio.TRIM)
        self.assertEqual(review(priced, holding, trim_premium=0.50).action, portfolio.HOLD)

    def test_weight_is_share_of_portfolio_value(self):
        result = review(assess(healthy(price=100.0)), Holding("X", quantity=10),
                        portfolio_value=5000.0)
        self.assertAlmostEqual(result.weight, 0.2)
        self.assertAlmostEqual(result.market_value, 1000.0)

    def test_unrealised_is_omitted_when_cost_basis_is_unknown(self):
        """A watchlist import has no cost basis; reporting 0% would read as break-even."""
        result = review(assess(healthy()), Holding("X", quantity=10))
        self.assertIsNone(result.unrealised_pct)


class SessionDetection(unittest.TestCase):
    def test_absent_session_is_reported_not_guessed(self):
        """A signed-out Yahoo page returns HTTP 200, so an empty parse is indistinguishable
        from an empty portfolio. The session check must gate the fetch."""
        ok, message = portfolio.session_status()
        self.assertIsInstance(ok, bool)
        self.assertTrue(message)
        if not ok:
            with self.assertRaises(PermissionError):
                portfolio.fetch_from_yahoo()

    def test_holdings_are_walked_out_of_nested_json(self):
        """Yahoo nests portfolios differently between layouts, so the parser looks for the
        shape rather than a fixed path."""
        found: dict[str, Holding] = {}
        portfolio._walk_for_holdings(
            {"context": {"lists": [{"items": [
                {"symbol": "T", "quantity": 500, "purchasePrice": 18.2},
                {"symbol": "KO"},
            ]}]}}, found)
        self.assertEqual(sorted(found), ["KO", "T"])
        self.assertEqual(found["T"].quantity, 500)

    def test_lowercase_keys_are_not_mistaken_for_tickers(self):
        """Yahoo's JSON is full of {"symbol": "..."} lookalikes; requiring an uppercase
        short string keeps prose and ids out of the portfolio."""
        found: dict[str, Holding] = {}
        portfolio._walk_for_holdings(
            {"symbol": "some-internal-identifier-value"}, found)
        self.assertEqual(found, {})


if __name__ == "__main__":
    unittest.main()
