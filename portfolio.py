"""Importing a Yahoo Finance portfolio and screening what is already owned.

Two sources, because Yahoo offers no public portfolio API and the private one needs a
logged-in session:

  * `cookies` -- reuse the Yahoo session from a local Firefox profile. Firefox stores
    cookies unencrypted, so an existing login can be reused without ever handling a
    password. Chrome encrypts its store with a per-user key, which is why this path is
    Firefox-only.
  * `csv` -- the file Yahoo's own Portfolio -> Export button produces. No account access
    at all, and it carries cost basis and quantity, which the web watchlist does not.

Owning a share asks a different question than buying one. A BUY screen wants a margin of
safety; a holder wants to know what has stopped deserving its place. So a held position
that screens EXPENSIVE or AVOID is surfaced as TRIM or EXIT, and the unrealised gain is
shown next to it -- a position up 300% that now trades at twice its estimated worth is
the single most common thing a buy-only screener never tells you.
"""

import csv
import glob
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

FIREFOX_COOKIE_GLOBS = (
    "~/snap/firefox/common/.mozilla/firefox/*/cookies.sqlite",
    "~/.mozilla/firefox/*/cookies.sqlite",
    "~/.var/app/org.mozilla.firefox/.mozilla/firefox/*/cookies.sqlite",
)

# Yahoo's session is carried by these; without them the portfolio endpoints return the
# signed-out payload rather than an error, which is worth detecting explicitly.
YAHOO_SESSION_COOKIES = ("SSL", "T", "Y", "A1", "A3")

# Header names Yahoo has used in its portfolio export. Matched case- and space-
# insensitively because the export has changed capitalisation between revisions.
SYMBOL_COLUMNS = ("symbol", "ticker")
QUANTITY_COLUMNS = ("quantity", "shares", "qty")
COST_COLUMNS = ("purchaseprice", "tradeprice", "costbasis", "purchase price")


@dataclass
class Holding:
    """One line of a portfolio. Quantity and cost are optional: a Yahoo *watchlist*
    exports symbols only, and that is still a usable screening input."""

    ticker: str
    quantity: float | None = None
    cost_basis: float | None = None

    @property
    def is_position(self) -> bool:
        """True when this is actually owned, rather than merely watched."""
        return bool(self.quantity and self.quantity > 0)

    def book_cost(self) -> float | None:
        if self.quantity and self.cost_basis:
            return self.quantity * self.cost_basis
        return None


def _normalise(header: str) -> str:
    return header.strip().lower().replace(" ", "").replace("_", "")


def _pick(row: dict, candidates: tuple[str, ...]) -> str | None:
    for key, value in row.items():
        if key and _normalise(key) in candidates:
            return value
    return None


def _to_float(value) -> float | None:
    """Yahoo exports numbers with thousands separators, currency marks and 'N/A'."""
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("$", "")
    if not text or text.upper() in ("N/A", "-", "NA", "NULL"):
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return None if result != result else result


def load_csv(path: Path) -> list[Holding]:
    """Parse a Yahoo portfolio/watchlist export.

    Columns are matched by name rather than position: Yahoo's export ordering differs
    between the portfolio and watchlist views, and an index-based parse silently reads
    the wrong column rather than failing.
    """
    holdings: list[Holding] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            symbol = (_pick(row, SYMBOL_COLUMNS) or "").strip().upper()
            # Yahoo appends a total row and blank separators to some exports.
            if not symbol or symbol.startswith("#") or symbol in ("TOTAL", "CASH"):
                continue
            holdings.append(Holding(
                ticker=symbol,
                quantity=_to_float(_pick(row, QUANTITY_COLUMNS)),
                cost_basis=_to_float(_pick(row, COST_COLUMNS)),
            ))
    if not holdings:
        raise ValueError(f"{path}: no ticker column found -- is this a Yahoo export?")
    return holdings


def firefox_cookie_stores() -> list[Path]:
    return [Path(p) for pattern in FIREFOX_COOKIE_GLOBS
            for p in glob.glob(os.path.expanduser(pattern))]


def yahoo_cookies() -> dict[str, str]:
    """Yahoo cookies from a local Firefox profile.

    The live database is copied before reading: Firefox holds a write lock while it is
    running, and opening it in place intermittently raises 'database is locked' rather
    than returning a partial result.
    """
    jar: dict[str, str] = {}
    for store in firefox_cookie_stores():
        with tempfile.TemporaryDirectory() as workdir:
            copy = Path(workdir) / "cookies.sqlite"
            try:
                shutil.copy2(store, copy)
                connection = sqlite3.connect(copy)
                rows = connection.execute(
                    "SELECT name, value FROM moz_cookies WHERE host LIKE '%yahoo.com%'"
                ).fetchall()
                connection.close()
            except (OSError, sqlite3.Error):
                continue
        jar.update({name: value for name, value in rows})
    return jar


def session_status() -> tuple[bool, str]:
    """Whether a reusable Yahoo login exists locally, and what to do if not."""
    stores = firefox_cookie_stores()
    if not stores:
        return False, ("No Firefox profile found. Chrome's cookie store is encrypted and "
                       "cannot be reused; export your portfolio to CSV instead.")
    jar = yahoo_cookies()
    if not jar:
        return False, ("Firefox has no Yahoo cookies: log in at finance.yahoo.com in "
                       "Firefox, then re-run.")
    if not any(name in jar for name in YAHOO_SESSION_COOKIES):
        return False, (f"Found {len(jar)} Yahoo cookies but none carrying a session. "
                       f"Log in at finance.yahoo.com in Firefox, then re-run.")
    return True, f"Yahoo session found ({len(jar)} cookies)."


PORTFOLIO_URL = "https://finance.yahoo.com/portfolios/"
BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0")


def _walk_for_holdings(node, out: dict[str, Holding]) -> None:
    """Collect every {symbol, quantity, purchase price} object anywhere in a JSON tree.

    Yahoo's page state nests portfolios differently between the classic and current
    layouts, and pinning an exact path means a silent empty result the next time they
    reshuffle it. Walking for the shape instead survives that.
    """
    if isinstance(node, dict):
        symbol = node.get("symbol") or node.get("ticker")
        if isinstance(symbol, str) and 0 < len(symbol) <= 12 and symbol.upper() == symbol:
            existing = out.get(symbol)
            quantity = _to_float(node.get("quantity") or node.get("shares"))
            cost = _to_float(node.get("purchasePrice") or node.get("costBasis")
                             or node.get("tradePrice"))
            # Keep the richest record: the same symbol appears in both a quote block
            # (price only) and a lot block (quantity and cost).
            if existing is None or (quantity and not existing.quantity):
                out[symbol] = Holding(symbol, quantity, cost)
        for value in node.values():
            _walk_for_holdings(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_for_holdings(value, out)


def fetch_from_yahoo(timeout: int = 30) -> list[Holding]:
    """Read the signed-in user's portfolios by reusing the local Firefox session.

    Raises PermissionError when no usable session exists, because the signed-out page
    returns HTTP 200 with a marketing body rather than a 401 -- parsing it would yield
    an empty portfolio indistinguishable from a genuinely empty one.
    """
    import json
    import re

    import requests

    ok, message = session_status()
    if not ok:
        raise PermissionError(message)

    response = requests.get(PORTFOLIO_URL, cookies=yahoo_cookies(),
                            headers={"User-Agent": BROWSER_UA}, timeout=timeout)
    response.raise_for_status()
    body = response.text

    if "login.yahoo.com" in body and "portfolios" not in body.lower():
        raise PermissionError("Yahoo served the signed-out page; the session has expired.")

    holdings: dict[str, Holding] = {}
    for match in re.finditer(r"root\.App\.main\s*=\s*(\{.*?\});\n", body, re.DOTALL):
        try:
            _walk_for_holdings(json.loads(match.group(1)), holdings)
        except json.JSONDecodeError:
            continue
    if not holdings:
        # Fall back to any embedded JSON blob on the page.
        for match in re.finditer(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',
                                 body, re.DOTALL):
            try:
                _walk_for_holdings(json.loads(match.group(1)), holdings)
            except json.JSONDecodeError:
                continue
    if not holdings:
        raise ValueError("Signed in, but no portfolio symbols were found on the page. "
                         "Use Yahoo's Portfolio -> Export and pass the CSV instead.")
    return sorted(holdings.values(), key=lambda h: h.ticker)


# --- Holdings-aware verdicts ----------------------------------------------------------
#
# A screen tells a buyer what to acquire. These translate the same assessment into what a
# holder should do with something already owned, which is not symmetrical: the threshold
# to sell is deliberately higher than the threshold to not-buy, because selling costs
# spread and tax while declining to buy costs nothing.

HOLD = "HOLD"
TRIM = "TRIM"
EXIT = "EXIT"
ADD = "ADD"

# How far above fair value a held position must trade before it is worth acting on.
TRIM_PREMIUM = 0.25


@dataclass
class PositionReview:
    ticker: str
    action: str
    market_value: float | None
    unrealised_pct: float | None
    weight: float | None
    rationale: str


def review(assessment, holding: Holding, portfolio_value: float | None = None,
           trim_premium: float = TRIM_PREMIUM) -> PositionReview:
    """What to do with a position already held.

    Quality failures exit regardless of price, matching the buy-side gates: the reason to
    own a business does not survive the business breaking, and a paper gain is not a
    reason to keep holding one that is burning cash.
    """
    import valuation

    price = assessment.price
    market_value = price * holding.quantity if holding.is_position else None
    unrealised = None
    if holding.cost_basis and holding.cost_basis > 0:
        unrealised = (price - holding.cost_basis) / holding.cost_basis
    weight = (market_value / portfolio_value
              if market_value and portfolio_value else None)

    if assessment.verdict == valuation.AVOID:
        action = EXIT
        rationale = "; ".join(assessment.gate_failures) or "failed quality gates"
    elif assessment.fair_value and price > assessment.fair_value * (1 + trim_premium):
        action = TRIM
        premium = price / assessment.fair_value - 1
        rationale = f"{premium:.0%} above estimated fair value"
    elif assessment.verdict == valuation.BUY:
        action = ADD
        rationale = f"still {assessment.margin_of_safety:.0%} below fair value"
    else:
        action = HOLD
        rationale = assessment.verdict.lower()

    return PositionReview(holding.ticker, action, market_value, unrealised, weight,
                          rationale)
