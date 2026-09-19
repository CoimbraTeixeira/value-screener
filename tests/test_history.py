"""Regression tests for the run history store.

Run with: python3 -m unittest discover -s tests

The point of storing runs is the derivative, not the snapshot, so these pin the
comparison logic: that a widening discount is correctly attributed to a price fall
rather than a rising estimate (opposite reactions, identical margin numbers), that a
ticker added to the watchlist today is not silently dropped for lacking a prior row,
and that comparison uses each ticker's own last reading rather than one global previous
run -- a watchlist is edited between runs, so those are not the same thing.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import history  # noqa: E402
import valuation  # noqa: E402
from history import Change, changes, connect, previous_run, record, run_count, trend  # noqa: E402
from valuation import assess  # noqa: E402

from test_valuation import agreeing, healthy  # noqa: E402


def store():
    """A throwaway history database."""
    path = Path(tempfile.mkdtemp()) / "history.db"
    return connect(path)


def change(**overrides) -> Change:
    base = dict(ticker="X", previous_at="2026-01-01T00:00:00+00:00",
                verdict_from=valuation.FAIR, verdict_to=valuation.FAIR,
                price_from=100.0, price_to=100.0, margin_from=0.0, margin_to=0.0)
    base.update(overrides)
    return Change(**base)


class Recording(unittest.TestCase):
    def test_a_run_round_trips(self):
        connection = store()
        first, second = assess(healthy()), assess(agreeing())
        second.ticker = "OTHER"
        stamp = record(connection, [first, second], run_at="2026-01-01T00:00:00+00:00")
        rows = connection.execute("SELECT * FROM snapshots ORDER BY ticker").fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["run_at"], stamp)
        self.assertEqual({row["ticker"] for row in rows}, {"TEST", "OTHER"})

    def test_one_ticker_twice_in_a_run_keeps_a_single_row(self):
        """A watchlist with a duplicate symbol should record it once, not twice at the
        same instant."""
        connection = store()
        record(connection, [assess(healthy()), assess(healthy())],
               run_at="2026-01-01T00:00:00+00:00")
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 1)

    def test_the_whole_batch_shares_one_timestamp(self):
        """Per-row times would make 'the previous run' ambiguous when a fetch of thirty
        tickers straddles a few seconds."""
        connection = store()
        first, second = assess(healthy()), assess(agreeing())
        second.ticker = "OTHER"
        record(connection, [first, second])
        stamps = connection.execute("SELECT DISTINCT run_at FROM snapshots").fetchall()
        self.assertEqual(len(stamps), 1)

    def test_rerunning_the_same_instant_replaces_rather_than_duplicates(self):
        connection = store()
        for _ in range(2):
            record(connection, [assess(healthy())], run_at="2026-01-01T00:00:00+00:00")
        self.assertEqual(run_count(connection), 1)
        self.assertEqual(
            connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 1)

    def test_an_empty_store_reports_no_runs(self):
        self.assertEqual(run_count(store()), 0)


class Comparison(unittest.TestCase):
    def test_a_ticker_with_no_prior_row_is_skipped_not_invented(self):
        """A symbol added to the watchlist today has nothing to compare against."""
        connection = store()
        self.assertEqual(changes(connection, [assess(healthy())], before="2030-01-01"), [])

    def test_comparison_uses_each_tickers_own_last_run(self):
        """Watchlists are edited. A symbol screened a month ago must still compare to
        that month-old reading rather than being dropped because a newer run of other
        tickers exists."""
        connection = store()
        old = assess(healthy(price=50.0))
        old.ticker = "OLD"
        record(connection, [old], run_at="2026-01-01T00:00:00+00:00")
        other = assess(healthy())
        other.ticker = "OTHER"
        record(connection, [other], run_at="2026-06-01T00:00:00+00:00")

        current = assess(healthy(price=80.0))
        current.ticker = "OLD"
        moved = changes(connection, [current], before="2026-12-01T00:00:00+00:00")
        self.assertEqual(len(moved), 1)
        self.assertEqual(moved[0].previous_at, "2026-01-01T00:00:00+00:00")

    def test_previous_run_ignores_rows_at_or_after_the_cutoff(self):
        """Otherwise a run compares against itself and nothing ever looks changed."""
        connection = store()
        record(connection, [assess(healthy())], run_at="2026-06-01T00:00:00+00:00")
        ticker = assess(healthy()).ticker
        self.assertIsNone(previous_run(connection, ticker, "2026-06-01T00:00:00+00:00"))
        self.assertIsNotNone(previous_run(connection, ticker, "2026-06-02T00:00:00+00:00"))


class Attribution(unittest.TestCase):
    def test_a_widening_discount_on_a_falling_price_is_named_as_such(self):
        """The market changed its mind about the price."""
        widened = change(margin_from=0.10, margin_to=0.30, price_from=100.0, price_to=80.0)
        self.assertIn("price fall", widened.describe())

    def test_a_widening_discount_on_a_flat_price_is_a_higher_estimate(self):
        """The business got better. Same margin move, opposite cause, opposite reaction."""
        widened = change(margin_from=0.10, margin_to=0.30, price_from=100.0, price_to=100.0)
        self.assertIn("higher estimate", widened.describe())

    def test_a_narrowing_discount_on_a_rising_price_is_named_as_such(self):
        narrowed = change(margin_from=0.30, margin_to=0.05, price_from=80.0, price_to=110.0)
        self.assertIn("price rise", narrowed.describe())

    def test_a_trivial_margin_move_is_not_dressed_up_as_news(self):
        self.assertEqual(change(margin_from=0.20, margin_to=0.205).describe(), "unchanged")

    def test_verdict_crossing_is_detected(self):
        self.assertTrue(change(verdict_from=valuation.FAIR,
                               verdict_to=valuation.BUY).verdict_changed)
        self.assertFalse(change().verdict_changed)

    def test_missing_margins_produce_no_attribution_rather_than_a_guess(self):
        """A stock that lost its anchors has no margin to compare."""
        self.assertIsNone(change(margin_to=None).margin_move)
        self.assertEqual(change(margin_to=None).describe(), "")


class Trend(unittest.TestCase):
    def test_history_reads_oldest_first(self):
        """A timeline printed newest-first reads backwards."""
        connection = store()
        for stamp, price in (("2026-01-01T00:00:00+00:00", 10.0),
                             ("2026-02-01T00:00:00+00:00", 20.0),
                             ("2026-03-01T00:00:00+00:00", 30.0)):
            record(connection, [assess(healthy(price=price))], run_at=stamp)
        prices = [row["price"] for row in trend(connection, assess(healthy()).ticker)]
        self.assertEqual(prices, [10.0, 20.0, 30.0])

    def test_limit_keeps_the_most_recent_runs(self):
        connection = store()
        for month in range(1, 6):
            record(connection, [assess(healthy(price=float(month)))],
                   run_at=f"2026-0{month}-01T00:00:00+00:00")
        rows = trend(connection, assess(healthy()).ticker, limit=2)
        self.assertEqual([row["price"] for row in rows], [4.0, 5.0])

    def test_unknown_ticker_yields_nothing(self):
        self.assertEqual(trend(store(), "NOSUCH"), [])


if __name__ == "__main__":
    unittest.main()
