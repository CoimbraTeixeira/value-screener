#!/usr/bin/env python3
"""Screen a watchlist of stocks for a margin of safety against an estimated fair value.

This is a screener, not an oracle. It does not know the correct price of anything. What
it does is apply four independent valuation models to each stock, refuse to act when
they disagree, veto on business quality before considering price at all, and show its
working so the output can be argued with. A BUY here means "this passed a mechanical
filter and is worth reading the filings for", nothing more.

  ./value_screener.py                     # screen watchlist.json
  ./value_screener.py AAPL MSFT KO        # screen specific tickers
  ./value_screener.py --explain AAPL      # show every anchor and why it abstained
  ./value_screener.py --only BUY,WATCH    # just the actionable rows
"""

import argparse
import json
import sys
from pathlib import Path

import portfolio
import valuation
from market_data import QUOTE_TTL_SECONDS, STATEMENT_TTL_SECONDS, fetch
from valuation import assess

REPO_DIR = Path(__file__).resolve().parent
# The real watchlist is gitignored: this repo is public and a tracked watchlist would
# publish which stocks its owner follows. A fresh clone falls back to the example so the
# screener runs out of the box.
WATCHLIST_PATH = REPO_DIR / "watchlist.json"
EXAMPLE_WATCHLIST_PATH = REPO_DIR / "watchlist.example.json"
CONFIG_PATH = Path.home() / ".config" / "fare-monitor" / "config.json"  # shared with the other monitors

# Ordered worst-to-best so a sort by index puts the actionable rows last, where the eye
# lands after reading a long table.
VERDICT_ORDER = [valuation.AVOID, valuation.NO_DATA, valuation.EXPENSIVE,
                 valuation.FAIR, valuation.WATCH, valuation.BUY]

VERDICT_MARK = {
    valuation.BUY: "++",
    valuation.WATCH: " +",
    valuation.FAIR: "  ",
    valuation.EXPENSIVE: " -",
    valuation.AVOID: " x",
    valuation.NO_DATA: " ?",
}


def load_watchlist(path: Path) -> list[str]:
    """Tickers from watchlist.json, accepting either a bare list or a {"tickers": [...]}
    object so the file can grow metadata later without breaking older runs."""
    if not path.exists():
        if not EXAMPLE_WATCHLIST_PATH.exists():
            sys.exit(f"No watchlist at {path}. Create one, or pass tickers as arguments.")
        print(f"# No {path.name} yet -- screening {EXAMPLE_WATCHLIST_PATH.name}. "
              f"Copy it to {path.name} and edit; yours stays out of git.",
              file=sys.stderr)
        path = EXAMPLE_WATCHLIST_PATH
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"Malformed JSON in {path}: {exc}")
    tickers = payload.get("tickers", []) if isinstance(payload, dict) else payload
    if not tickers:
        sys.exit(f"{path} lists no tickers.")
    return [str(t).upper().strip() for t in tickers]


def format_row(a: valuation.Assessment) -> str:
    fair = f"{a.fair_value:>10,.2f}" if a.fair_value else f"{'--':>10}"
    margin = f"{a.margin_of_safety:>7.0%}" if a.margin_of_safety is not None else f"{'--':>7}"
    reason = ""
    if a.gate_failures:
        reason = "; ".join(a.gate_failures)
    elif a.notes:
        reason = a.notes[0]
    return (f"{VERDICT_MARK[a.verdict]} {a.ticker:<6} {a.price:>10,.2f} {fair} {margin}  "
            f"{a.verdict:<9} {reason[:44]}")


def format_explanation(a: valuation.Assessment) -> str:
    """Per-anchor breakdown, including the abstentions.

    The refusals are printed alongside the numbers on purpose: knowing that three of
    four models declined to value a company is the most useful thing the screener can
    say about it, and hiding abstentions would make a one-anchor guess look like a
    consensus.
    """
    lines = [f"\n{a.ticker} - {a.name}", f"  price {a.price:,.2f} {a.currency}"]
    for anchor in a.anchors:
        if anchor.value is None:
            lines.append(f"  {anchor.name:<9} {'abstained':>12}  ({anchor.detail})")
        else:
            implied = (anchor.value - a.price) / anchor.value
            lines.append(f"  {anchor.name:<9} {anchor.value:>12,.2f}  "
                         f"{implied:>6.0%} vs price  ({anchor.detail})")
    if a.fair_value:
        lines.append(f"  {'median':<9} {a.fair_value:>12,.2f}  "
                     f"{a.margin_of_safety:>6.0%} margin of safety")
    if a.analyst_target:
        lines.append(f"  {'analysts':<9} {a.analyst_target:>12,.2f}  (consensus target, for contrast)")
    for failure in a.gate_failures:
        lines.append(f"  GATE      {failure}")
    for note in a.notes:
        lines.append(f"  NOTE      {note}")
    lines.append(f"  -> {a.verdict}")
    return "\n".join(lines)


def build_summary(results: list[valuation.Assessment], buy_margin: float) -> str:
    """The Discord message: only what changes a decision, plus the caveat."""
    actionable = [r for r in results if r.verdict in (valuation.BUY, valuation.WATCH)]
    if not actionable:
        return (f"Screened {len(results)} stocks: nothing at a "
                f"{buy_margin:.0%} margin of safety.")
    lines = [f"**Value screen** -- {len(actionable)} of {len(results)} stocks below "
             f"estimated fair value:"]
    for r in sorted(actionable, key=lambda x: -(x.margin_of_safety or 0)):
        lines.append(f"- **{r.ticker}** {r.price:,.2f} vs {r.fair_value:,.2f} est. "
                     f"({r.margin_of_safety:.0%} margin) - {r.verdict}")
    lines.append("_Estimates from four disagreeing models. Not advice; read the filings._")
    return "\n".join(lines)


def load_holdings(args) -> list[portfolio.Holding]:
    """Holdings from whichever source the flags name. Empty list means watchlist mode."""
    if args.portfolio:
        try:
            return portfolio.load_csv(Path(args.portfolio))
        except (OSError, ValueError) as exc:
            sys.exit(f"Could not read {args.portfolio}: {exc}")
    if args.from_yahoo:
        try:
            return portfolio.fetch_from_yahoo()
        except PermissionError as exc:
            sys.exit(f"No Yahoo session: {exc}")
        except Exception as exc:
            sys.exit(f"Yahoo portfolio fetch failed: {exc}")
    return []


def format_positions(results: list[valuation.Assessment],
                     holdings: list[portfolio.Holding],
                     all_results: list[valuation.Assessment] | None = None) -> str:
    """The holder's view: what to do with what is already owned.

    Printed as a second table rather than extra columns on the first, because the buy
    question and the hold question have different answers for the same stock -- a
    position can be simultaneously EXPENSIVE (do not buy more) and worth holding.
    """
    by_ticker = {r.ticker: r for r in results}
    owned = [h for h in holdings if h.is_position and h.ticker in by_ticker]
    if not owned:
        return ""

    # Weights are a share of every position held, not of the rows that survived --only.
    priced = {r.ticker: r for r in (all_results or results)}
    total = sum(priced[h.ticker].price * h.quantity
                for h in holdings if h.is_position and h.ticker in priced)
    reviews = [portfolio.review(by_ticker[h.ticker], h, total) for h in owned]
    order = {portfolio.EXIT: 0, portfolio.TRIM: 1, portfolio.HOLD: 2, portfolio.ADD: 3}
    reviews.sort(key=lambda r: (order[r.action], -(r.market_value or 0)))

    lines = ["", f"{'HOLDINGS':<8} {'VALUE':>12} {'WEIGHT':>7} {'P/L':>8}  "
                 f"{'ACTION':<6} WHY"]
    for r in reviews:
        weight = f"{r.weight:>6.1%}" if r.weight is not None else f"{'--':>6}"
        pnl = f"{r.unrealised_pct:>7.0%}" if r.unrealised_pct is not None else f"{'--':>7}"
        lines.append(f"{r.ticker:<8} {r.market_value:>12,.0f} {weight} {pnl}  "
                     f"{r.action:<6} {r.rationale[:42]}")
    lines.append(f"{'TOTAL':<8} {total:>12,.0f}")

    flagged = [r for r in reviews if r.action in (portfolio.EXIT, portfolio.TRIM)]
    if flagged:
        at_risk = sum(r.market_value for r in flagged)
        lines.append(f"\n{len(flagged)} position(s) flagged, {at_risk / total:.0%} of "
                     f"portfolio value. Cost basis and tax are not considered here.")
    return "\n".join(lines)


def notify(message: str) -> None:
    import requests

    if not CONFIG_PATH.exists():
        sys.exit(f"Missing config file: {CONFIG_PATH}")
    url = json.loads(CONFIG_PATH.read_text()).get("webhook_url")
    if not url:
        sys.exit(f"No 'webhook_url' key in {CONFIG_PATH}")
    response = requests.post(url, json={"content": message}, timeout=30)
    response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", help="Tickers to screen (default: watchlist.json)")
    parser.add_argument("--explain", action="store_true",
                        help="Show every anchor per stock, including abstentions")
    parser.add_argument("--only", default="",
                        help="Comma-separated verdicts to show, e.g. BUY,WATCH")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="Emit machine-readable results instead of a table")
    parser.add_argument("--notify", action="store_true", help="Post a summary to Discord")
    parser.add_argument("--buy-margin", type=float, default=valuation.BUY_MARGIN,
                        help=f"Margin of safety required for BUY (default "
                             f"{valuation.BUY_MARGIN:.0%})")
    parser.add_argument("--risk-free", type=float, default=valuation.DEFAULT_RISK_FREE,
                        help="Risk-free rate used in the discount rate")
    parser.add_argument("--equity-premium", type=float, default=valuation.DEFAULT_EQUITY_PREMIUM,
                        help="Equity risk premium used in the discount rate")
    parser.add_argument("--terminal-growth", type=float, default=valuation.DEFAULT_TERMINAL_GROWTH,
                        help="Perpetual growth rate after year 10")
    parser.add_argument("--max-debt-to-equity", type=float, default=valuation.MAX_DEBT_TO_EQUITY,
                        help="Leverage gate, in percent (200 = 2.0x equity)")
    parser.add_argument("--no-cache", action="store_true", help="Refetch everything")
    parser.add_argument("--portfolio", metavar="CSV",
                        help="Screen a Yahoo Finance portfolio/watchlist export instead "
                             "of watchlist.json")
    parser.add_argument("--from-yahoo", action="store_true",
                        help="Screen the portfolio from your signed-in Yahoo session "
                             "(reuses a local Firefox login; no password needed)")
    parser.add_argument("--yahoo-status", action="store_true",
                        help="Report whether a reusable Yahoo login was found, and exit")
    args = parser.parse_args()

    if args.yahoo_status:
        ok, message = portfolio.session_status()
        print(("OK   " if ok else "NONE ") + message)
        sys.exit(0 if ok else 1)

    holdings = load_holdings(args)
    tickers = ([t.upper() for t in args.tickers]
               or [h.ticker for h in holdings]
               or load_watchlist(WATCHLIST_PATH))
    quote_ttl = 0 if args.no_cache else QUOTE_TTL_SECONDS
    statement_ttl = 0 if args.no_cache else STATEMENT_TTL_SECONDS

    results, failures = [], []
    for ticker in tickers:
        try:
            fundamentals = fetch(ticker, quote_ttl=quote_ttl, statement_ttl=statement_ttl)
        except Exception as exc:
            # One unknown ticker must not abort a 30-stock run; collect and report.
            failures.append(f"{ticker}: {exc}")
            continue
        results.append(assess(
            fundamentals,
            risk_free=args.risk_free,
            equity_premium=args.equity_premium,
            terminal_growth=args.terminal_growth,
            max_debt_to_equity=args.max_debt_to_equity,
            buy_margin=args.buy_margin,
        ))

    # Position weights must be a share of the whole portfolio, so --only filters the
    # screening table without shrinking the denominator underneath the holdings table.
    screened = results
    wanted = {v.strip().upper() for v in args.only.split(",") if v.strip()}
    if wanted:
        results = [r for r in results if r.verdict in wanted]
    results.sort(key=lambda r: (VERDICT_ORDER.index(r.verdict), r.margin_of_safety or -9))

    if args.as_json:
        print(json.dumps([{
            "ticker": r.ticker, "name": r.name, "price": r.price, "currency": r.currency,
            "verdict": r.verdict, "fair_value": r.fair_value,
            "margin_of_safety": r.margin_of_safety,
            "anchors": {a.name: a.value for a in r.anchors},
            "gate_failures": r.gate_failures, "notes": r.notes,
        } for r in results], indent=2))
    elif args.explain:
        for r in results:
            print(format_explanation(r))
    else:
        print(f"   {'TICKER':<6} {'PRICE':>10} {'FAIR EST':>10} {'MARGIN':>7}  "
              f"{'VERDICT':<9} NOTE")
        for r in results:
            print(format_row(r))
        if any(h.is_position for h in holdings):
            print(format_positions(results, holdings, screened))

    for failure in failures:
        print(f" ! {failure}", file=sys.stderr)

    if not args.as_json:
        counts = {v: sum(1 for r in results if r.verdict == v) for v in VERDICT_ORDER}
        print(f"\n{len(results)} screened: "
              + ", ".join(f"{n} {v.lower()}" for v, n in counts.items() if n))
        print("Fair values are estimates from disagreeing models, not price targets.")

    if args.notify:
        notify(build_summary(results, args.buy_margin))


if __name__ == "__main__":
    main()
