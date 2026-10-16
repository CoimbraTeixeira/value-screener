"""Tests for provider normalisation in market_data.

Run with: python3 -m unittest discover -s tests

The payment series are real Yahoo data. Yahoo dropped `dividendRate` from the quote in
October 2026 and every dividend anchor went blank while the morning post kept going
out, so the annual rate is now derived from the payments themselves.
"""

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_data  # noqa: E402
from market_data import annual_dividend  # noqa: E402

AS_OF = date(2026, 10, 16)

KO = [["2025-06-13", 0.51], ["2025-09-15", 0.51], ["2025-12-01", 0.51],
      ["2026-03-13", 0.53], ["2026-06-15", 0.53], ["2026-09-15", 0.53]]
NOVO = [["2024-08-15", 3.5], ["2025-03-28", 7.9], ["2025-08-15", 3.75],
        ["2026-03-27", 7.95], ["2026-08-14", 3.75]]
INTC = [["2023-11-06", 0.125], ["2024-02-06", 0.125], ["2024-05-06", 0.125],
        ["2024-08-07", 0.125]]
O = [["2026-05-29", 0.271], ["2026-06-30", 0.271], ["2026-07-31", 0.271],
     ["2026-08-31", 0.271], ["2026-09-30", 0.272]]


class AnnualDividend(unittest.TestCase):
    def test_a_quarterly_raise_counts_from_the_first_raised_payment(self):
        """KO raised to 0.53: Yahoo's old rate read 2.12, the trailing sum only 2.08."""
        self.assertAlmostEqual(annual_dividend(KO, AS_OF), 2.12)

    def test_monthly_payers_annualise_twelve_times(self):
        self.assertAlmostEqual(annual_dividend(O, AS_OF), 0.272 * 12)

    def test_uneven_interim_and_final_payments_are_summed(self):
        """Doubling Novo's 3.75 interim would understate its 11.70 by a third."""
        self.assertAlmostEqual(annual_dividend(NOVO, AS_OF), 3.75 + 7.95)

    def test_a_suspended_dividend_is_no_dividend(self):
        """Intel's last payment was August 2024; annualising it would invent a yield."""
        self.assertIsNone(annual_dividend(INTC, AS_OF))

    def test_one_payment_has_no_readable_cadence(self):
        self.assertIsNone(annual_dividend([["2026-09-15", 0.53]], AS_OF))


class Normalisation(unittest.TestCase):
    QUOTE = {"currentPrice": 92.0, "fetched_at": "2026-10-16T13:30:00Z", "dividends": KO}

    def test_the_rate_comes_from_payments_when_yahoo_omits_it(self):
        f = market_data._fundamentals("KO", self.QUOTE, {}, None)
        self.assertAlmostEqual(f.dividend_rate, 2.12)
        self.assertAlmostEqual(f.dividend_yield, 2.12 / 92.0)

    def test_yahoos_rate_wins_when_present(self):
        f = market_data._fundamentals("KO", dict(self.QUOTE, dividendRate=2.04), {}, None)
        self.assertEqual(f.dividend_rate, 2.04)

    def test_staleness_is_judged_at_fetch_time_not_replay_time(self):
        """A recording replayed a year later must value as it did the day it was made."""
        f = market_data._fundamentals("INTC", {"currentPrice": 30.0, "dividends": INTC,
                                               "fetched_at": "2024-09-01T13:30:00Z"},
                                      {}, None)
        self.assertAlmostEqual(f.dividend_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
