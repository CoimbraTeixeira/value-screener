#!/usr/bin/env python3
"""Post the day's screen to Discord at the opening bell, on days the market opens.

Scheduled by cron across a window rather than at a fixed hour, because this machine
runs on Europe/Dublin and the exchange on America/New_York, and the two switch daylight
saving on different dates -- a line pinned to 14:30 local is an hour wrong for about
four weeks a year. Cron polls; market_clock decides. See the footer of this file for the
crontab line.

Two guards keep a polling schedule from becoming a spamming one: the run aborts unless
the moment really is just after the bell on a trading day, and a state file records the
exchange date already posted so a second tick in the same window is a no-op.

The message leads with what *changed*. A list of verdicts is the same most mornings and
gets skimmed into invisibility; a crossing into BUY, or a holding that just failed a
quality gate, is the reason to have opened the notification at all.
"""

import argparse
import fcntl
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import history
import market_clock
import valuation
from market_data import fetch
from valuation import assess
from value_screener import WATCHLIST_PATH, load_watchlist, notify

REPO_DIR = Path(__file__).resolve().parent
STATE_DIR = REPO_DIR / "state"
STATE_PATH = STATE_DIR / "market_open_posted.json"
LOCK_PATH = STATE_DIR / "market_open.lock"

# Discord rejects a body over 2000 characters outright rather than truncating it, so the
# message is built to a budget instead of hoping it fits.
DISCORD_LIMIT = 2000

# How long after the bell a run still counts as "at the open". Generous enough to
# absorb cron granularity, a slow fetch or a missed tick; short enough that a machine
# woken at lunchtime does not post a stale "market open" note.
DEFAULT_TOLERANCE_MINUTES = 45


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}")


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        # A truncated state file would otherwise block every future run. Losing the
        # record costs one duplicate post; refusing to run costs every post after it.
        log(f"State file {STATE_PATH} is unreadable; treating as empty.")
        return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE_PATH)


def already_posted(state: dict, session: str) -> bool:
    return state.get("last_session") == session


def screen(tickers: list[str]) -> tuple[list, list[str]]:
    results, failures = [], []
    for ticker in tickers:
        try:
            results.append(assess(fetch(ticker)))
        except Exception as exc:
            # One dead ticker must not cost the whole morning's message.
            failures.append(f"{ticker}: {exc}")
    return results, failures


def build_message(results: list, changes: list, session: str,
                  failures: list[str]) -> str:
    """The morning post: what changed first, then what is currently actionable."""
    actionable = sorted(
        (r for r in results if r.verdict in (valuation.BUY, valuation.WATCH)),
        key=lambda r: -(r.margin_of_safety or 0))
    crossings = [c for c in changes if c.verdict_changed]

    lines = [f"**Market open** - {session} - {len(results)} screened"]

    if crossings:
        lines.append("")
        lines.append("__Changed since last run__")
        for change in crossings:
            lines.append(f"- **{change.ticker}** {change.verdict_from} -> "
                         f"{change.verdict_to} ({change.describe()})")

    if actionable:
        lines.append("")
        lines.append("__Below estimated fair value__")
        for r in actionable:
            flags = f" - {r.flags[0]}" if r.flags else ""
            lines.append(f"- **{r.ticker}** {r.price:,.2f} vs {r.fair_value:,.2f} est. "
                         f"({r.margin_of_safety:.0%}) {r.verdict}{flags}")
    else:
        lines.append("")
        lines.append("Nothing below estimated fair value today.")

    gated = [r for r in results if r.verdict == valuation.AVOID]
    if gated:
        lines.append("")
        lines.append(f"_{len(gated)} failing quality gates: "
                     + ", ".join(r.ticker for r in gated[:8]) + "_")

    if failures:
        lines.append(f"_{len(failures)} could not be fetched._")

    lines.append("")
    lines.append("_Estimates from disagreeing models. Not advice._")

    message = "\n".join(lines)
    if len(message) > DISCORD_LIMIT:
        # Trim the actionable list rather than the changes: the changes are the news.
        message = message[:DISCORD_LIMIT - 40].rsplit("\n", 1)[0] + "\n_(truncated)_"
    return message


def parse_when(text: str) -> datetime:
    """An ISO timestamp for --now, rejected unless it carries a zone.

    A naive value would be read as UTC and put the simulated bell an hour out, which is
    exactly the class of bug this flag exists to catch.
    """
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        sys.exit(f"--now: not an ISO 8601 timestamp: {text!r}")
    if moment.tzinfo is None:
        sys.exit("--now: include a timezone offset, e.g. 2026-09-21T13:30:00+00:00")
    return moment


def run(args) -> int:
    # A time-gated job that can only be exercised at 14:30 on a weekday is a job whose
    # scheduling is never tested until it silently fails. --now makes the gate reachable.
    now = parse_when(args.now) if args.now else datetime.now(timezone.utc)
    elapsed = market_clock.minutes_since_open(now)
    local = market_clock.exchange_now(now)
    session = local.date().isoformat()

    if not args.force:
        if elapsed is None:
            log(f"{session} is not a trading day (weekend or NYSE holiday). "
                f"Next open {market_clock.next_open(now):%Y-%m-%d %H:%M %Z}.")
            return 0
        if not market_clock.is_opening_window(now, args.tolerance):
            log(f"Not the opening window: {elapsed:+.0f} min from the bell "
                f"(tolerance {args.tolerance}). Nothing to do.")
            return 0

    state = load_state()
    if already_posted(state, session) and not args.force:
        log(f"Already posted for {session}.")
        return 0

    tickers = load_watchlist(WATCHLIST_PATH)
    log(f"Screening {len(tickers)} tickers for {session}.")
    results, failures = screen(tickers)
    if not results:
        log("No results; not posting.")
        return 1

    connection = history.connect()
    stamp = now.isoformat(timespec="seconds")
    changes = history.changes(connection, results, before=stamp)
    history.record(connection, results, run_at=stamp)
    connection.close()

    message = build_message(results, changes, session, failures)

    if args.dry_run:
        print(message)
        return 0

    notify(message)
    save_state({"last_session": session, "posted_at": stamp})
    log(f"Posted {len(results)} results for {session}.")
    for failure in failures:
        log(f"  fetch failed: {failure}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="Ignore the market-hours and once-a-day guards")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the message instead of posting it")
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE_MINUTES,
                        help=f"Minutes after the bell still counted as the open "
                             f"(default {DEFAULT_TOLERANCE_MINUTES})")
    parser.add_argument("--now", metavar="ISO8601",
                        help="Pretend it is this moment, to exercise the market-hours "
                             "gate without waiting for the bell (e.g. "
                             "2026-09-21T13:30:00+00:00)")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            # A slow run must be skipped, not queued: two overlapping runs would post
            # the same morning twice.
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("Another run holds the lock; skipping this tick.")
            sys.exit(0)
        sys.exit(run(args))


if __name__ == "__main__":
    main()

# Crontab: poll every 15 minutes across both possible local hours for the bell, and let
# market_clock decide which tick is the real one. 13:xx and 14:xx Dublin cover 09:30 New
# York on either side of the daylight-saving mismatch.
#
#   */15 13,14 * * 1-5 /usr/bin/python3 \
#     /home/rogerio/Documents/repos/value-screener/market_open_monitor.py \
#     >> /home/rogerio/Documents/repos/value-screener/market_open.log 2>&1
