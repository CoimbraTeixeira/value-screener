"""Regression tests for the forward-looking layer.

Run with: python3 -m unittest discover -s tests

Forward data is the easiest way to make a screener confidently wrong, so these pin the
guards rather than the plumbing: a stock split read as 280% dilution, a single analyst's
guess weighted like four years of filings, routine option-exercise selling reported as
an insider signal, and -- the one that costs real money -- a large discount treated as a
finding while the earnings underneath it are being revised away.

The Apple split case is a real number: yfinance's raw share series reads 4.4bn for 2019
against 17bn for 2020 purely because of the 4:1 split.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valuation  # noqa: E402
from market_data import MAX_PLAUSIBLE_SHARE_CHANGE, _build_forward  # noqa: E402
from valuation import (  # noqa: E402
    Forward,
    assess,
    dcf_anchor,
    estimate_growth,
    estimates_falling,
    forward_flags,
    forward_pe_anchor,
)

from test_valuation import agreeing, healthy  # noqa: E402


def covered(**overrides) -> Forward:
    """Forward data with enough analyst coverage to be usable."""
    base = dict(eps_next_year=9.0, analyst_count=20, revisions_up=5, revisions_down=2)
    base.update(overrides)
    return Forward(**base)


class Dilution(unittest.TestCase):
    def test_issuing_shares_lowers_value_per_share(self):
        """Identical cash flows split across more shares are worth less each. This is
        arithmetic, not a forecast, which is why it moves the number rather than raising
        a flag."""
        flat = dcf_anchor(healthy(forward=Forward(share_growth=0.0))).value
        diluting = dcf_anchor(healthy(forward=Forward(share_growth=0.05))).value
        self.assertLess(diluting, flat)

    def test_buybacks_raise_value_per_share(self):
        flat = dcf_anchor(healthy(forward=Forward(share_growth=0.0))).value
        shrinking = dcf_anchor(healthy(forward=Forward(share_growth=-0.03))).value
        self.assertGreater(shrinking, flat)

    def test_absent_forward_data_matches_no_dilution(self):
        """The forward layer must be additive: a stock with no estimates available has to
        value exactly as it did before the layer existed."""
        self.assertAlmostEqual(dcf_anchor(healthy()).value,
                               dcf_anchor(healthy(forward=Forward(share_growth=0.0))).value)

    def test_extreme_share_growth_is_clamped(self):
        """A stock-funded acquisition is not a ten-year issuance policy."""
        absurd = dcf_anchor(healthy(forward=Forward(share_growth=3.0))).value
        clamped = dcf_anchor(
            healthy(forward=Forward(share_growth=valuation.MAX_SHARE_GROWTH))).value
        self.assertAlmostEqual(absurd, clamped)

    def test_dilution_is_disclosed_in_the_detail(self):
        self.assertIn("dilution", dcf_anchor(healthy(forward=Forward(share_growth=0.04))).detail)
        self.assertIn("buyback", dcf_anchor(healthy(forward=Forward(share_growth=-0.04))).detail)

    def test_split_sized_jumps_are_rejected_as_issuance(self):
        """Apple's 4:1 split appears as a 4x share count jump. The threshold has to sit
        below that and above any real issuance programme."""
        apple_split_ratio = 17_001_799_680 / 4_443_270_144 - 1.0
        self.assertGreater(apple_split_ratio, MAX_PLAUSIBLE_SHARE_CHANGE)
        self.assertGreater(MAX_PLAUSIBLE_SHARE_CHANGE, 0.10)


class ForwardEpsAnchor(unittest.TestCase):
    def test_abstains_on_thin_coverage(self):
        """Two analysts are an opinion. Weighting that like four years of filings is how
        a small cap gets a confident fair value from one person's spreadsheet."""
        anchor = forward_pe_anchor(healthy(forward=covered(analyst_count=2)))
        self.assertIsNone(anchor.value)
        self.assertIn("analysts covering", anchor.detail)

    def test_abstains_without_estimates(self):
        self.assertIsNone(forward_pe_anchor(healthy()).value)
        self.assertIsNone(forward_pe_anchor(healthy(forward=covered(eps_next_year=None))).value)

    def test_abstains_on_a_forecast_loss(self):
        self.assertIsNone(forward_pe_anchor(healthy(forward=covered(eps_next_year=-1.0))).value)

    def test_future_earnings_are_discounted_back_a_year(self):
        """A value arriving twelve months out is not worth its face amount today."""
        f = healthy(forward=covered(eps_next_year=8.0), historical_pe=[10.0, 10.0])
        rate = valuation.discount_rate(f.beta)
        self.assertAlmostEqual(forward_pe_anchor(f).value, 10.0 * 8.0 / (1 + rate))

    def test_it_can_see_a_recovery_that_trailing_anchors_cannot(self):
        """The reason this anchor exists: a company earning little today but expected to
        recover is condemned by every backward-looking model here."""
        depressed = healthy(eps_trailing=0.5, historical_pe=[15.0, 15.0],
                            forward=covered(eps_next_year=8.0))
        self.assertGreater(forward_pe_anchor(depressed).value,
                           valuation.historical_pe_anchor(depressed).value)


class GrowthBlending(unittest.TestCase):
    def test_forward_estimates_can_only_lower_growth(self):
        """Sell-side forecasts are persistently optimistic, so they join the min() rather
        than replacing history. A rosy forecast must not raise the DCF."""
        history_only = estimate_growth(healthy(earnings_growth=0.05))
        with_rosy = estimate_growth(healthy(earnings_growth=0.05,
                                            forward=Forward(long_term_growth=0.40)))
        self.assertAlmostEqual(history_only, with_rosy)

    def test_a_pessimistic_forecast_does_lower_growth(self):
        base = estimate_growth(healthy(earnings_growth=0.05))
        lowered = estimate_growth(healthy(earnings_growth=0.05,
                                          forward=Forward(revenue_growth_next_year=0.01)))
        self.assertLess(lowered, base)


class EstimateMomentum(unittest.TestCase):
    def test_revision_balance_detects_cuts(self):
        self.assertTrue(estimates_falling(Forward(revisions_up=1, revisions_down=9)))
        self.assertFalse(estimates_falling(Forward(revisions_up=9, revisions_down=1)))

    def test_slow_drift_is_caught_even_without_revision_counts(self):
        """A consensus ground down over 90 days never shows up as a dramatic week."""
        self.assertTrue(estimates_falling(Forward(eps_drift_90d=-0.10)))
        self.assertFalse(estimates_falling(Forward(eps_drift_90d=0.05)))

    def test_no_forward_data_is_not_a_falling_estimate(self):
        self.assertFalse(estimates_falling(None))
        self.assertFalse(estimates_falling(Forward()))

    def test_a_balanced_split_is_not_a_cut(self):
        self.assertFalse(estimates_falling(Forward(revisions_up=5, revisions_down=5)))


class ValueTrapVeto(unittest.TestCase):
    def test_cheap_with_falling_estimates_is_not_a_buy(self):
        """The expensive mistake this whole layer exists to prevent: the price fell
        because the earnings are about to, and every trailing anchor is still pricing
        earnings that are disappearing."""
        trap = agreeing(forward=Forward(revisions_up=0, revisions_down=8))
        result = assess(trap)
        self.assertGreater(result.margin_of_safety, valuation.BUY_MARGIN)
        self.assertEqual(result.verdict, valuation.WATCH)

    def test_the_same_stock_with_rising_estimates_is_a_buy(self):
        """Controls for everything except revision direction."""
        healthy_estimates = agreeing(forward=Forward(revisions_up=8, revisions_down=0))
        self.assertEqual(assess(healthy_estimates).verdict, valuation.BUY)

    def test_the_veto_does_not_manufacture_a_discount(self):
        """Flags must not touch the fair value, only the confidence in acting on it."""
        without = assess(agreeing())
        with_cuts = assess(agreeing(forward=Forward(revisions_up=0, revisions_down=8)))
        self.assertAlmostEqual(without.fair_value, with_cuts.fair_value)


class Flags(unittest.TestCase):
    def test_routine_insider_selling_is_not_reported(self):
        """Option exercises and 10b5-1 plans make insider selling near-universal; a flag
        that fires on every stock carries no information."""
        f = healthy(shares_outstanding=1.0e8,
                    forward=Forward(insider_net_shares_6m=-50_000))
        self.assertEqual([x for x in forward_flags(f) if "insider" in x], [])

    def test_insider_buying_is_reported(self):
        f = healthy(shares_outstanding=1.0e8, forward=Forward(insider_net_shares_6m=200_000))
        self.assertTrue(any("net buyers" in x for x in forward_flags(f)))

    def test_material_selling_is_reported(self):
        f = healthy(shares_outstanding=1.0e8, forward=Forward(insider_net_shares_6m=-2.0e6))
        self.assertTrue(any("heavy insider selling" in x for x in forward_flags(f)))

    def test_imminent_earnings_are_flagged(self):
        self.assertTrue(any("reports in" in x
                            for x in forward_flags(healthy(forward=Forward(days_to_earnings=3)))))
        self.assertFalse(any("reports in" in x
                             for x in forward_flags(healthy(forward=Forward(days_to_earnings=45)))))

    def test_dilution_is_flagged_as_well_as_priced(self):
        self.assertTrue(any("diluting" in x
                            for x in forward_flags(healthy(forward=Forward(share_growth=0.06)))))

    def test_no_forward_data_produces_no_flags(self):
        self.assertEqual(forward_flags(healthy()), [])


class AnalystTargets(unittest.TestCase):
    def test_price_targets_never_enter_the_fair_value(self):
        """Targets track the current price with a lag. Letting them into the estimate
        would launder consensus into a number whose purpose is to disagree with it."""
        low = assess(healthy(analyst_target=1.0))
        high = assess(healthy(analyst_target=10_000.0))
        self.assertAlmostEqual(low.fair_value, high.fair_value)

    def test_a_wildly_different_target_is_shown_as_a_contrast(self):
        result = assess(healthy(analyst_target=10_000.0))
        self.assertTrue(any("analyst target" in n for n in result.notes))


class CacheCompatibility(unittest.TestCase):
    def test_unknown_cached_keys_are_ignored(self):
        """The forward cache outlives a run by a day, so a payload written by a newer
        version must not crash an older one."""
        built = _build_forward({"eps_next_year": 5.0, "some_future_field": 123})
        self.assertEqual(built.eps_next_year, 5.0)

    def test_empty_payload_yields_no_forward(self):
        self.assertIsNone(_build_forward(None))
        self.assertIsNone(_build_forward({}))


if __name__ == "__main__":
    unittest.main()
