"""When the US market is open, computed rather than guessed.

Two things here are easy to get quietly wrong.

The first is daylight saving. This machine runs on Europe/Dublin and the exchange on
America/New_York, and the two do not switch on the same dates: the US springs forward on
the second Sunday in March while the EU waits until the last Sunday, and the EU falls
back a week before the US does. For roughly four weeks a year the offset is four hours
rather than five, so a cron line pinned to 14:30 local fires an hour late every spring
and an hour early every autumn. Everything below works in exchange-local time and lets
zoneinfo resolve the offset.

The second is the holiday calendar. A screener that posts "market open" on Thanksgiving
is worse than one that posts nothing, because it teaches you to ignore it. The NYSE
calendar is rule-based and therefore computable -- no network call, no dependency, and
testable against dates that have already happened.

Deliberately not modelled: early closes. The half-days after Thanksgiving and before
Christmas still open at 09:30, and this module only ever answers questions about the
opening bell.
"""

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

EXCHANGE_TZ = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)

# Juneteenth became an NYSE holiday in 2022; before that the market traded.
JUNETEENTH_FROM = 2022


def easter(year: int) -> date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher).

    Needed only because Good Friday is the one NYSE holiday with no fixed date and no
    nth-weekday rule -- it is the single reason this module does arithmetic instead of
    reading a table.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lunar = (32 + 2 * e + 2 * i - h - k) % 7
    month_offset = (a + 11 * h + 22 * lunar) // 451
    month, day = divmod(h + lunar - 7 * month_offset + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The nth given weekday of a month, e.g. the 3rd Monday of January."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    """The last given weekday of a month, e.g. the last Monday of May."""
    next_month = date(year + month // 12, month % 12 + 1, 1)
    last = next_month - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed(holiday: date) -> date | None:
    """Where a fixed-date holiday actually falls when it lands on a weekend.

    The NYSE shifts a Saturday holiday back to Friday and a Sunday holiday forward to
    Monday. New Year's Day is the exception and is handled by the caller: a Saturday
    January 1st would shift into the previous year, so the exchange simply trades
    through the preceding Friday.
    """
    if holiday.weekday() == 5:
        return holiday - timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + timedelta(days=1)
    return holiday


def market_holidays(year: int) -> set[date]:
    """Every full NYSE closure in a calendar year."""
    holidays = {
        nth_weekday(year, 1, 0, 3),            # Martin Luther King Jr Day
        nth_weekday(year, 2, 0, 3),            # Washington's Birthday
        easter(year) - timedelta(days=2),      # Good Friday
        last_weekday(year, 5, 0),              # Memorial Day
        nth_weekday(year, 9, 0, 1),            # Labor Day
        nth_weekday(year, 11, 3, 4),           # Thanksgiving
    }

    # New Year's Day: observed on Monday when it falls on Sunday, but a Saturday 1st is
    # not made up on the preceding Friday, which belongs to the previous year.
    new_year = date(year, 1, 1)
    if new_year.weekday() != 5:
        holidays.add(observed(new_year))
    # A Sunday 31st December pushes the following year's holiday into the second, but a
    # Saturday 1st of January the *following* year closes nothing in this one.

    for fixed in (date(year, 7, 4), date(year, 12, 25)):
        holidays.add(observed(fixed))
    if year >= JUNETEENTH_FROM:
        holidays.add(observed(date(year, 6, 19)))

    return {day for day in holidays if day.year == year}


def is_trading_day(day: date) -> bool:
    """True when the NYSE holds a regular session."""
    return day.weekday() < 5 and day not in market_holidays(day.year)


def exchange_now(now: datetime | None = None) -> datetime:
    """The current moment in exchange-local time, whatever this machine runs on."""
    moment = now or datetime.now(tz=EXCHANGE_TZ)
    if moment.tzinfo is None:
        raise ValueError("a naive datetime cannot be converted to exchange time")
    return moment.astimezone(EXCHANGE_TZ)


def minutes_since_open(now: datetime | None = None) -> float | None:
    """Minutes since today's opening bell, or None when today is not a trading day.

    Negative before the bell. Returned as a signed number rather than a boolean so the
    caller can choose its own tolerance, and so a run that fires early is distinguishable
    from one that fires late.
    """
    local = exchange_now(now)
    if not is_trading_day(local.date()):
        return None
    bell = local.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                         second=0, microsecond=0)
    return (local - bell).total_seconds() / 60.0


def is_opening_window(now: datetime | None = None, tolerance_minutes: int = 45) -> bool:
    """Whether now is within the window just after the opening bell.

    A window rather than an instant because cron granularity, a slow fetch and a missed
    tick all shift the actual run time. The window starts *at* the bell: firing before
    the open would report yesterday's closing prices as though they were today's.
    """
    elapsed = minutes_since_open(now)
    return elapsed is not None and 0 <= elapsed <= tolerance_minutes


def next_open(now: datetime | None = None) -> datetime:
    """The next opening bell, in exchange-local time."""
    local = exchange_now(now)
    candidate = local.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                              second=0, microsecond=0)
    if candidate <= local or not is_trading_day(candidate.date()):
        candidate += timedelta(days=1)
        while not is_trading_day(candidate.date()):
            candidate += timedelta(days=1)
        candidate = candidate.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute)
    return candidate
