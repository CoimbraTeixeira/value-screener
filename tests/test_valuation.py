"""Regression tests for the valuation engine.

Run with: python3 -m unittest discover -s tests

These cover the ways a screener produces a confident wrong number rather than no number:
a growth rate invented from a negative base, a Gordon model dividing by a negative
denominator, a NaN sliding through the thresholds into whichever branch is last, and a
large apparent discount computed from anchors that flatly contradict each other. Each
test pins a decision the output depends on, not the shape of the code.

The MSFT free-cash-flow case is here because it shipped broken: Yahoo's summary field
reported 16.5bn against a filed 67bn and the DCF valued the share at a sixth of its
price, with nothing in the output to suggest the input was wrong.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valuation  # noqa: E402
from market_data import _number, _series  # noqa: E402
from valuation import (  # noqa: E402
    Anchor,
    Fundamentals,
    assess,
    cagr,
    dcf_anchor,
    discount_rate,
    dispersion,
    dividend_anchor,
    estimate_growth,
    graham_anchor,
    historical_pe_anchor,
)


def healthy(**overrides) -> Fundamentals:
    """A profitable, cash-generative, modestly-levered company at a plausible price."""
    base = dict(
        ticker="TEST", price=100.0, name="Test Corp",
        eps_trailing=8.0, book_value_per_share=40.0,
        free_cash_flow=1.0e9, shares_outstanding=1.0e8,
        total_cash=2.0e8, total_debt=1.0e8, beta=1.0,
        dividend_rate=3.0, dividend_yield=0.03, payout_ratio=0.4,
        return_on_equity=0.18, debt_to_equity=60.0, earnings_growth=0.05,
        fcf_history=[8.0e8, 8.6e8, 9.2e8, 1.0e9],
        historical_pe=[14.0, 15.0, 16.0, 15.5],
    )
    base.update(overrides)
    return Fundamentals(**base)


def agreeing(**overrides) -> Fundamentals:
    """A company whose four anchors land within 1.3x of each other.

    Deliberately separate from healthy(): that fixture's anchors sit 2.6x apart, which is
    the normal state of affairs and exercises the downgrade path. Testing the BUY verdict
    needs a case where the models genuinely concur, otherwise the test passes or fails on
    the dispersion rule rather than on the margin of safety.
    """
    base = dict(price=40.0, historical_pe=[11.0, 11.0],
                free_cash_flow=4.5e8, fcf_history=[3.8e8, 4.5e8])
    base.update(overrides)
    return healthy(**base)


class GrowthEstimation(unittest.TestCase):
    def test_cagr_refuses_a_negative_starting_point(self):
        """A run from -50 to 100 has no compound rate. Returning one would feed the DCF a
        fictional growth number derived from a sign flip."""
        self.assertIsNone(cagr([-50.0, 100.0]))
        self.assertIsNone(cagr([50.0, -100.0]))
        self.assertIsNone(cagr([100.0]))

    def test_cagr_computes_over_periods_not_points(self):
        # 100 -> 133.1 across three intervals is 10%, not 33% or 7.5%.
        self.assertAlmostEqual(cagr([100.0, 110.0, 121.0, 133.1]), 0.10, places=6)

    def test_growth_takes_the_lower_of_history_and_forecast(self):
        """The conservative input wins, because it is the one not contingent on a forecast."""
        f = healthy(fcf_history=[100.0, 200.0], earnings_growth=0.03)  # history says 100%
        self.assertAlmostEqual(estimate_growth(f), 0.03)

    def test_growth_is_capped_and_floored(self):
        soaring = healthy(fcf_history=[100.0, 900.0], earnings_growth=0.95)
        self.assertEqual(estimate_growth(soaring), valuation.MAX_STAGE1_GROWTH)
        shrinking = healthy(fcf_history=[900.0, 100.0], earnings_growth=-0.4)
        self.assertEqual(estimate_growth(shrinking), valuation.MIN_STAGE1_GROWTH)

    def test_growth_defaults_to_zero_when_nothing_is_known(self):
        self.assertEqual(estimate_growth(healthy(fcf_history=[], earnings_growth=None)), 0.0)


class DiscountRate(unittest.TestCase):
    def test_absurd_beta_is_clamped(self):
        """A thin-float stock printing beta 4 would otherwise imply a 24% discount rate and
        a near-zero DCF driven entirely by that artefact."""
        self.assertEqual(discount_rate(4.0), valuation.MAX_DISCOUNT)
        self.assertEqual(discount_rate(0.01), valuation.MIN_DISCOUNT)

    def test_missing_beta_assumes_the_market(self):
        self.assertAlmostEqual(discount_rate(None), discount_rate(1.0))


class DcfAnchor(unittest.TestCase):
    def test_abstains_rather_than_dividing_by_a_negative_denominator(self):
        """Terminal growth above the discount rate makes Gordon growth diverge; the model
        must decline instead of returning the negative value the formula yields."""
        anchor = dcf_anchor(healthy(), terminal_growth=0.20)
        self.assertIsNone(anchor.value)
        self.assertIn("terminal growth", anchor.detail)

    def test_abstains_on_negative_cash_flow(self):
        self.assertIsNone(dcf_anchor(healthy(free_cash_flow=-5.0e8)).value)

    def test_abstains_when_net_debt_swamps_the_cash_stream(self):
        anchor = dcf_anchor(healthy(total_debt=1.0e12, total_cash=0.0))
        self.assertIsNone(anchor.value)
        self.assertIn("net debt", anchor.detail)

    def test_net_cash_lifts_the_value_by_its_per_share_amount(self):
        without = dcf_anchor(healthy(total_cash=0.0, total_debt=0.0)).value
        with_cash = dcf_anchor(healthy(total_cash=1.0e9, total_debt=0.0)).value
        self.assertAlmostEqual(with_cash - without, 1.0e9 / 1.0e8, places=6)

    def test_higher_discount_rate_lowers_the_value(self):
        cheap_money = dcf_anchor(healthy(), risk_free=0.0).value
        dear_money = dcf_anchor(healthy(), risk_free=0.10).value
        self.assertLess(dear_money, cheap_money)

    def test_a_zero_growth_dcf_still_values_the_existing_stream(self):
        """The floor case must be a real number, not zero: a no-growth business is worth
        its cash flows, and returning nothing here would make the median drop an anchor."""
        flat = healthy(fcf_history=[1.0e9, 1.0e9], earnings_growth=0.0)
        value = dcf_anchor(flat).value
        self.assertIsNotNone(value)
        self.assertGreater(value, 0.0)


class HistoricalPeAnchor(unittest.TestCase):
    def test_uses_the_median_so_one_collapsed_year_cannot_dominate(self):
        """A year where EPS nearly vanished prints a 232x multiple -- seen on real GILD
        data. The median must ignore it; a mean would nearly double the fair value."""
        f = healthy(historical_pe=[17.8, 232.3, 17.0, 20.8], eps_trailing=5.0)
        self.assertAlmostEqual(historical_pe_anchor(f).value, 19.3 * 5.0, places=6)

    def test_bubble_multiple_is_capped_and_the_cap_is_disclosed(self):
        f = healthy(historical_pe=[80.0, 90.0], eps_trailing=5.0)
        anchor = historical_pe_anchor(f)
        self.assertAlmostEqual(anchor.value, valuation.DEFAULT_MAX_ANCHOR_PE * 5.0)
        self.assertIn("capped", anchor.detail)

    def test_abstains_without_enough_history(self):
        self.assertIsNone(historical_pe_anchor(healthy(historical_pe=[15.0])).value)

    def test_abstains_on_a_loss(self):
        self.assertIsNone(historical_pe_anchor(healthy(eps_trailing=-1.0)).value)


class GrahamAnchor(unittest.TestCase):
    def test_matches_the_published_formula(self):
        f = healthy(eps_trailing=8.0, book_value_per_share=40.0)
        self.assertAlmostEqual(graham_anchor(f).value, math.sqrt(22.5 * 8.0 * 40.0))

    def test_abstains_on_negative_book_value(self):
        self.assertIsNone(graham_anchor(healthy(book_value_per_share=-3.0)).value)


class DividendAnchor(unittest.TestCase):
    def test_abstains_on_an_immaterial_yield(self):
        """A 0.3% yield says nothing about what the share is worth, and a Gordon model on
        it would still emit a confident number."""
        self.assertIsNone(dividend_anchor(healthy(dividend_yield=0.003)).value)

    def test_abstains_when_there_is_no_dividend(self):
        self.assertIsNone(dividend_anchor(healthy(dividend_rate=None)).value)

    def test_growth_cannot_reach_the_discount_rate(self):
        """ROE 60% with full retention would imply growth far above any discount rate and
        a negative denominator; the cap must hold."""
        f = healthy(return_on_equity=0.60, payout_ratio=0.0, beta=0.2)
        anchor = dividend_anchor(f)
        self.assertTrue(anchor.value is None or anchor.value > 0)


class Gates(unittest.TestCase):
    def test_quality_failure_outranks_a_large_discount(self):
        """A stock at a third of its estimated worth is still AVOID if the business is
        burning cash. Price is the second question."""
        result = assess(healthy(price=10.0, free_cash_flow=-1.0e9))
        self.assertEqual(result.verdict, valuation.AVOID)
        self.assertTrue(any("burning cash" in f for f in result.gate_failures))

    def test_missing_data_is_not_a_gate_failure(self):
        """An absent field is a gap in the feed, not evidence of a bad business."""
        blank = healthy(free_cash_flow=None, return_on_equity=None, debt_to_equity=None,
                        eps_trailing=None)
        self.assertEqual(valuation.check_gates(blank), [])

    def test_leverage_gate_respects_its_threshold(self):
        self.assertEqual(valuation.check_gates(healthy(debt_to_equity=199.0)), [])
        self.assertEqual(len(valuation.check_gates(healthy(debt_to_equity=201.0))), 1)

    def test_banks_and_reits_are_not_condemned_for_negative_cash_flow(self):
        """Equinix builds data centres and JPMorgan lends money, so both report negative
        free cash flow and high leverage through entirely healthy decades. Applying
        those two gates there produced confident false negatives on whole sectors."""
        for sector in ("Financial Services", "Real Estate"):
            wrongly_condemned = healthy(sector=sector, free_cash_flow=-5.0e9,
                                        debt_to_equity=400.0)
            self.assertEqual(valuation.check_gates(wrongly_condemned), [], sector)

    def test_exempt_sectors_are_still_judged_on_profitability(self):
        """The exemption is narrow: a bank that has stopped earning is still AVOID."""
        failures = valuation.check_gates(
            healthy(sector="Financial Services", eps_trailing=-2.0))
        self.assertTrue(any("unprofitable" in f for f in failures))

    def test_ordinary_sectors_keep_the_cash_flow_gate(self):
        failures = valuation.check_gates(healthy(sector="Technology", free_cash_flow=-1.0))
        self.assertTrue(any("burning cash" in f for f in failures))

    def test_condemned_company_with_no_anchors_says_avoid_not_no_data(self):
        """Intel screened as NO DATA while unprofitable and cash-burning -- the facts that
        removed the anchors were themselves the verdict."""
        wreck = Fundamentals(ticker="X", price=50.0, eps_trailing=-3.0,
                             free_cash_flow=-1.0e9, shares_outstanding=1.0e8)
        result = assess(wreck)
        self.assertEqual(result.verdict, valuation.AVOID)


class Verdicts(unittest.TestCase):
    def test_wide_disagreement_downgrades_a_buy_to_a_watch(self):
        """A 30% discount computed from anchors 5x apart is noise with a decimal point."""
        spread = healthy(price=10.0, historical_pe=[15.0, 15.0], eps_trailing=8.0,
                         book_value_per_share=0.5)
        result = assess(spread)
        self.assertGreater(dispersion(result.usable_anchors), valuation.MAX_TRUSTED_DISPERSION)
        self.assertEqual(result.verdict, valuation.WATCH)
        self.assertTrue(any("disagree" in n for n in result.notes))

    def test_a_clear_discount_with_agreeing_anchors_is_a_buy(self):
        result = assess(agreeing())
        self.assertLessEqual(dispersion(result.usable_anchors),
                             valuation.MAX_TRUSTED_DISPERSION)
        self.assertEqual(result.verdict, valuation.BUY)
        self.assertGreater(result.margin_of_safety, valuation.BUY_MARGIN)

    def test_buy_margin_is_configurable(self):
        """Same stock, stricter threshold: the knob must be what decides, so this uses the
        agreeing fixture rather than one already downgraded for dispersion."""
        self.assertEqual(assess(agreeing()).verdict, valuation.BUY)
        self.assertNotEqual(assess(agreeing(), buy_margin=0.95).verdict, valuation.BUY)

    def test_expensive_when_price_exceeds_every_anchor(self):
        self.assertEqual(assess(healthy(price=100000.0)).verdict, valuation.EXPENSIVE)

    def test_funds_are_refused_outright(self):
        """An ETF has no earnings or book value of its own; every anchor would be
        measuring its holdings by accident."""
        result = assess(healthy(quote_type="ETF"))
        self.assertEqual(result.verdict, valuation.NO_DATA)
        self.assertIsNone(result.fair_value)

    def test_fair_value_is_the_median_not_the_mean(self):
        """One anchor blowing up must not drag the estimate: the median is what survives a
        single bad input."""
        result = assess(healthy())
        values = sorted(a.value for a in result.usable_anchors)
        self.assertIn(len(values), (3, 4))
        self.assertLessEqual(values[0], result.fair_value)
        self.assertLessEqual(result.fair_value, values[-1])

    def test_margin_of_safety_is_measured_against_fair_value(self):
        result = assess(healthy(price=50.0))
        self.assertAlmostEqual(
            result.margin_of_safety,
            (result.fair_value - 50.0) / result.fair_value, places=9)


class ProviderNormalisation(unittest.TestCase):
    def test_nan_is_rejected(self):
        """pandas hands back NaN for missing rows, and NaN compares False against every
        threshold, so it lands in whichever branch happens to be last."""
        self.assertIsNone(_number(float("nan")))
        self.assertIsNone(_number(None))
        self.assertIsNone(_number("not a number"))
        self.assertEqual(_number("12.5"), 12.5)

    def test_booleans_are_not_numbers(self):
        self.assertIsNone(_number(True))

    def test_series_is_oldest_first_with_gaps_dropped(self):
        import pandas as pd

        frame = pd.DataFrame(
            {"2025": [3.0], "2024": [2.0], "2023": [float("nan")]}, index=["Free Cash Flow"])
        self.assertEqual(_series(frame, "Free Cash Flow"), [2.0, 3.0])

    def test_series_tolerates_a_missing_row(self):
        import pandas as pd

        frame = pd.DataFrame({"2025": [1.0]}, index=["Operating Cash Flow"])
        self.assertEqual(_series(frame, "Free Cash Flow"), [])
        self.assertEqual(_series(None, "Free Cash Flow"), [])


class Dispersion(unittest.TestCase):
    def test_needs_two_values(self):
        self.assertIsNone(dispersion([Anchor("a", 10.0, ""), Anchor("b", None, "")]))

    def test_is_the_ratio_of_extremes(self):
        anchors = [Anchor("a", 10.0, ""), Anchor("b", 30.0, ""), Anchor("c", 20.0, "")]
        self.assertAlmostEqual(dispersion(anchors), 3.0)


if __name__ == "__main__":
    unittest.main()
