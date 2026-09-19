"""Regression tests for the market calendar and the opening-bell window.

Run with: python3 -m unittest discover -s tests

Two failure modes are pinned here because both are silent.

The daylight-saving one: this machine runs on Europe/Dublin and the exchange on
America/New_York, and they switch on different dates. For about four weeks a year the
offset is four hours rather than five, so anything pinned to a local clock time fires an
hour wrong -- and an alert an hour late still looks plausible, which is what makes it
dangerous.

The calendar one: posting "market open" on Thanksgiving trains the reader to ignore the
notification. Holidays are asserted against dates that have already happened, so these
are checkable against the published NYSE calendar rather than against the code's own
opinion.
"""

import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import market_clock  # noqa: E402
from market_clock import (  # noqa: E402
    easter,
    is_opening_window,
    is_trading_day,
    last_weekday,
    market_holidays,
    minutes_since_open,
    next_open,
    nth_weekday,
)

DUBLIN = ZoneInfo("Europe/Dublin")
NEW_YORK = ZoneInfo("America/New_York")


class Easter(unittest.TestCase):
    def test_known_easters(self):
        """Good Friday is the only NYSE holiday with no fixed date and no nth-weekday
        rule, so the whole calendar rests on this arithmetic being right."""
        self.assertEqual(easter(2024), date(2024, 3, 31))
        self.assertEqual(easter(2025), date(2025, 4, 20))
        self.assertEqual(easter(2026), date(2026, 4, 5))
        self.assertEqual(easter(2027), date(2027, 3, 28))

    def test_good_friday_is_a_closure(self):
        self.assertIn(date(2025, 4, 18), market_holidays(2025))
        self.assertFalse(is_trading_day(date(2026, 4, 3)))


class WeekdayHelpers(unittest.TestCase):
    def test_nth_weekday(self):
        self.assertEqual(nth_weekday(2026, 1, 0, 3), date(2026, 1, 19))   # MLK
        self.assertEqual(nth_weekday(2026, 11, 3, 4), date(2026, 11, 26))  # Thanksgiving

    def test_last_weekday_handles_december_rollover(self):
        """The naive next-month calculation overflows the year here."""
        self.assertEqual(last_weekday(2026, 12, 0), date(2026, 12, 28))
        self.assertEqual(last_weekday(2026, 5, 0), date(2026, 5, 25))     # Memorial Day


class Holidays(unittest.TestCase):
    def test_2025_matches_the_published_nyse_calendar(self):
        self.assertEqual(market_holidays(2025), {
            date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
            date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
            date(2025, 11, 27), date(2025, 12, 25)})

    def test_a_saturday_holiday_is_observed_on_the_friday(self):
        """4 July 2026 is a Saturday, so the exchange closes Friday the 3rd."""
        self.assertIn(date(2026, 7, 3), market_holidays(2026))
        self.assertFalse(is_trading_day(date(2026, 7, 3)))

    def test_a_sunday_holiday_is_observed_on_the_monday(self):
        """Juneteenth 2027 falls on a Saturday; Christmas 2028 on a Monday. Check the
        Sunday case explicitly: 25 December 2022 was a Sunday, observed the 26th."""
        self.assertIn(date(2022, 12, 26), market_holidays(2022))

    def test_a_saturday_new_year_closes_nothing(self):
        """The exception to the Saturday rule: shifting back would land in the previous
        year, so the exchange simply trades through 31 December."""
        self.assertTrue(is_trading_day(date(2027, 12, 31)))
        self.assertNotIn(date(2027, 12, 31), market_holidays(2027))

    def test_juneteenth_only_after_it_became_a_holiday(self):
        self.assertNotIn(date(2021, 6, 18), market_holidays(2021))
        self.assertIn(date(2022, 6, 20), market_holidays(2022))

    def test_weekends_are_not_trading_days(self):
        self.assertFalse(is_trading_day(date(2026, 9, 19)))  # Saturday
        self.assertFalse(is_trading_day(date(2026, 9, 20)))  # Sunday
        self.assertTrue(is_trading_day(date(2026, 9, 21)))   # Monday


class DaylightSaving(unittest.TestCase):
    def test_the_bell_is_1430_dublin_when_both_zones_agree(self):
        moment = datetime(2026, 9, 21, 14, 30, tzinfo=DUBLIN)
        self.assertAlmostEqual(minutes_since_open(moment), 0.0)
        self.assertTrue(is_opening_window(moment))

    def test_the_bell_is_1330_dublin_inside_the_march_mismatch(self):
        """The US springs forward on the second Sunday in March, the EU on the last.
        Between those dates the offset is four hours, not five."""
        moment = datetime(2026, 3, 17, 13, 30, tzinfo=DUBLIN)
        self.assertAlmostEqual(minutes_since_open(moment), 0.0)
        self.assertTrue(is_opening_window(moment))

    def test_a_fixed_local_hour_would_be_an_hour_late_in_march(self):
        """This is precisely the bug the window exists to prevent."""
        moment = datetime(2026, 3, 17, 14, 30, tzinfo=DUBLIN)
        self.assertAlmostEqual(minutes_since_open(moment), 60.0)
        self.assertFalse(is_opening_window(moment, tolerance_minutes=45))

    def test_utc_input_is_converted_not_assumed(self):
        from datetime import timezone

        moment = datetime(2026, 9, 21, 13, 30, tzinfo=timezone.utc)  # 09:30 EDT
        self.assertAlmostEqual(minutes_since_open(moment), 0.0)

    def test_a_naive_datetime_is_refused(self):
        """Silently assuming a zone is how an alert ends up an hour out."""
        with self.assertRaises(ValueError):
            minutes_since_open(datetime(2026, 9, 21, 9, 30))


class OpeningWindow(unittest.TestCase):
    def test_it_does_not_fire_before_the_bell(self):
        """Firing early would report yesterday's closing prices as today's."""
        moment = datetime(2026, 9, 21, 9, 25, tzinfo=NEW_YORK)
        self.assertFalse(is_opening_window(moment))
        self.assertLess(minutes_since_open(moment), 0)

    def test_it_fires_across_the_tolerance_and_stops_after(self):
        inside = datetime(2026, 9, 21, 10, 10, tzinfo=NEW_YORK)
        outside = datetime(2026, 9, 21, 10, 30, tzinfo=NEW_YORK)
        self.assertTrue(is_opening_window(inside, tolerance_minutes=45))
        self.assertFalse(is_opening_window(outside, tolerance_minutes=45))

    def test_a_holiday_never_opens_however_right_the_clock_looks(self):
        thanksgiving = datetime(2026, 11, 26, 9, 30, tzinfo=NEW_YORK)
        self.assertIsNone(minutes_since_open(thanksgiving))
        self.assertFalse(is_opening_window(thanksgiving))

    def test_a_weekend_never_opens(self):
        saturday = datetime(2026, 9, 19, 9, 30, tzinfo=NEW_YORK)
        self.assertFalse(is_opening_window(saturday))


class NextOpen(unittest.TestCase):
    def test_friday_evening_points_at_monday(self):
        friday = datetime(2026, 9, 18, 18, 0, tzinfo=NEW_YORK)
        self.assertEqual(next_open(friday).date(), date(2026, 9, 21))

    def test_it_skips_a_holiday(self):
        """The Wednesday before Thanksgiving should point at the Friday."""
        wednesday = datetime(2026, 11, 25, 18, 0, tzinfo=NEW_YORK)
        self.assertEqual(next_open(wednesday).date(), date(2026, 11, 27))

    def test_before_the_bell_points_at_today(self):
        early = datetime(2026, 9, 21, 7, 0, tzinfo=NEW_YORK)
        self.assertEqual(next_open(early).date(), date(2026, 9, 21))

    def test_it_always_lands_on_the_bell(self):
        moment = datetime(2026, 9, 18, 18, 0, tzinfo=NEW_YORK)
        opening = next_open(moment)
        self.assertEqual((opening.hour, opening.minute), (9, 30))
        self.assertTrue(is_trading_day(opening.date()))


class PostedOnce(unittest.TestCase):
    def test_a_session_is_recorded_and_recognised(self):
        """Cron polls a window, so the same bell is seen several times; only the first
        may post."""
        import market_open_monitor as monitor

        self.assertTrue(monitor.already_posted({"last_session": "2026-09-21"},
                                               "2026-09-21"))
        self.assertFalse(monitor.already_posted({"last_session": "2026-09-18"},
                                                "2026-09-21"))
        self.assertFalse(monitor.already_posted({}, "2026-09-21"))

    def test_a_zoned_override_is_parsed(self):
        """--now exists so the market-hours gate can be exercised without waiting for
        a weekday bell; a job whose scheduling is only testable at 14:30 on a Monday is
        a job that fails silently."""
        import market_open_monitor as monitor

        parsed = monitor.parse_when("2026-09-21T13:30:00+00:00")
        self.assertEqual(market_clock.exchange_now(parsed).hour, 9)

    def test_a_naive_override_is_refused(self):
        """Reading it as UTC would put the simulated bell an hour out, which is the bug
        the flag exists to catch."""
        import market_open_monitor as monitor

        with self.assertRaises(SystemExit):
            monitor.parse_when("2026-09-21T13:30:00")

    def test_nonsense_is_refused(self):
        import market_open_monitor as monitor

        with self.assertRaises(SystemExit):
            monitor.parse_when("tomorrow morning")


if __name__ == "__main__":
    unittest.main()
