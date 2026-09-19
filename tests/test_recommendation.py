"""Regression tests for the plain buy/hold/sell call and the Discord message.

Run with: python3 -m unittest discover -s tests

Two things are pinned. The recommendation mapping, because collapsing six verdicts into
three words is where nuance gets lost in the wrong direction -- a cheap price must never
outvote a failed quality gate, and mild overvaluation must not read as a sell when the
estimate's own error bar is wider than the gap.

And message splitting, because Discord rejects an over-long body with an HTTP 400
rather than truncating it. A watchlist that grew past thirty-odd rows would silently
stop posting altogether, which is the worst failure mode available to a notification.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import valuation  # noqa: E402
from valuation import assess, recommendation  # noqa: E402
from value_screener import DISCORD_LIMIT, split_message  # noqa: E402

from test_valuation import agreeing, healthy  # noqa: E402


class Mapping(unittest.TestCase):
    def test_a_clear_discount_is_a_buy(self):
        action, reason = recommendation(assess(agreeing()))
        self.assertEqual(action, valuation.RECOMMEND_BUY)
        self.assertIn("below estimate", reason)

    def test_a_failed_gate_is_a_sell_however_cheap(self):
        """The gates fire on unprofitable and cash-burning businesses. A low price is
        not a reason to own one, so price must not outvote the gate."""
        broken = assess(healthy(price=1.0, free_cash_flow=-1.0e9))
        action, reason = recommendation(broken)
        self.assertEqual(action, valuation.RECOMMEND_SELL)
        self.assertIn("burning cash", reason)

    def test_many_gate_failures_are_summarised_not_listed(self):
        """The full text runs past 150 characters on the worst names, and a business
        failing four gates is not four times as informative as one failing a single."""
        wreck = assess(healthy(price=10.0, eps_trailing=-1.0, free_cash_flow=-1.0,
                               return_on_equity=-0.5, debt_to_equity=900.0))
        _, reason = recommendation(wreck)
        self.assertIn("more)", reason)
        self.assertLess(len(reason), 80)

    def test_mild_overvaluation_is_a_hold_not_a_sell(self):
        """Churning on a gap narrower than the estimate's own error costs spread for
        nothing."""
        base = assess(healthy())
        mild = assess(healthy(price=base.fair_value * 1.10))
        self.assertEqual(recommendation(mild)[0], valuation.RECOMMEND_HOLD)

    def test_large_overvaluation_is_a_sell(self):
        base = assess(healthy())
        rich = assess(healthy(price=base.fair_value * 1.60))
        action, reason = recommendation(rich)
        self.assertEqual(action, valuation.RECOMMEND_SELL)
        self.assertIn("above estimate", reason)

    def test_the_sell_threshold_matches_the_portfolio_trim_threshold(self):
        """The same gap must not mean TRIM in the holdings table and HOLD here."""
        import portfolio

        self.assertEqual(valuation.SELL_PREMIUM, portfolio.TRIM_PREMIUM)

    def test_a_fund_gets_no_call(self):
        action, reason = recommendation(assess(healthy(quote_type="ETF")))
        self.assertEqual(action, valuation.RECOMMEND_NONE)
        self.assertIn("do not apply", reason)

    def test_a_blocked_buy_holds_and_says_why(self):
        """A large discount undermined by falling estimates is a HOLD, and the reason
        shown should be the blocker rather than the margin."""
        trap = agreeing(forward=valuation.Forward(revisions_up=0, revisions_down=8))
        result = assess(trap)
        self.assertEqual(result.verdict, valuation.WATCH)
        action, reason = recommendation(result)
        self.assertEqual(action, valuation.RECOMMEND_HOLD)
        self.assertTrue(reason)

    def test_every_verdict_maps_to_something(self):
        """No verdict may fall through to an empty call."""
        cases = [healthy(), agreeing(), healthy(price=1e6), healthy(quote_type="ETF"),
                 healthy(eps_trailing=-1.0, free_cash_flow=-1.0)]
        for fundamentals in cases:
            action, reason = recommendation(assess(fundamentals))
            self.assertIn(action, {valuation.RECOMMEND_BUY, valuation.RECOMMEND_HOLD,
                                   valuation.RECOMMEND_SELL, valuation.RECOMMEND_NONE})
            self.assertTrue(reason)


class Splitting(unittest.TestCase):
    def test_a_short_message_is_not_split(self):
        self.assertEqual(split_message("one\ntwo"), ["one\ntwo"])

    def test_chunks_respect_the_limit(self):
        text = "\n".join(f"line {n}" for n in range(500))
        for chunk in split_message(text, limit=200):
            self.assertLessEqual(len(chunk), 200)

    def test_no_line_is_lost(self):
        """Truncating instead would drop whichever stocks happened to sort last."""
        lines = [f"`TICK{n:<4}` {n:>9,.2f} - some reason here" for n in range(60)]
        text = "\n".join(lines)
        rejoined = "\n".join(split_message(text, limit=300))
        for line in lines:
            self.assertIn(line, rejoined)

    def test_a_single_overlong_line_is_cut_rather_than_looping(self):
        """Cannot happen with the rows built here, but an unbounded loop would hang the
        cron job rather than fail it."""
        chunks = split_message("x" * 5000, limit=1000)
        self.assertEqual(len(chunks), 5)

    def test_the_real_limit_is_discords(self):
        self.assertEqual(DISCORD_LIMIT, 2000)


class MessageShape(unittest.TestCase):
    def build(self, results):
        import market_open_monitor as monitor

        return monitor.build_message(results, [], "2026-09-21", [])

    def test_every_screened_stock_appears(self):
        """The whole point of the change: a call on all of them, not just the
        actionable ones."""
        first, second, third = assess(healthy()), assess(agreeing()), assess(
            healthy(quote_type="ETF"))
        second.ticker, third.ticker = "SECOND", "THIRD"
        message = self.build([first, second, third])
        for ticker in ("TEST", "SECOND", "THIRD"):
            self.assertIn(ticker, message)

    def test_groups_are_counted(self):
        message = self.build([assess(agreeing())])
        self.assertIn("__BUY (1)__", message)

    def test_the_sell_caveat_is_always_present(self):
        """Without a cost basis, SELL cannot mean 'close your position', and the message
        must not imply that it does."""
        self.assertIn("not worth owning", self.build([assess(healthy())]))

    def test_empty_groups_are_omitted(self):
        message = self.build([assess(agreeing())])
        self.assertNotIn("HOLD (0)", message)
        self.assertNotIn("No call", message)


if __name__ == "__main__":
    unittest.main()
