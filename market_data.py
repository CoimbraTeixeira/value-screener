"""Fetching and normalising fundamentals from Yahoo via yfinance.

Kept separate from valuation.py so the maths can be tested without a network, and so
provider quirks are fixed in exactly one place. yfinance's field names and units drift
between releases, and the normalisation here exists to stop that drift reaching the
models.

Caching is two-tier because the two halves of a fetch have completely different
lifetimes. Quote data moves every minute; annual statements move four times a year and
cost three extra HTTP round trips each. One TTL for both would either serve stale prices
or refetch five years of financials to learn that a stock moved twelve cents.
"""

import json
import os
import time
from pathlib import Path

import yfinance as yf

from valuation import Fundamentals

REPO_DIR = Path(__file__).resolve().parent
CACHE_DIR = REPO_DIR / "cache"

QUOTE_TTL_SECONDS = 3600            # 1 hour
STATEMENT_TTL_SECONDS = 7 * 86400   # 7 days

# Yahoo returns fiscal period ends; a market close on the exact date may not exist
# (weekends, holidays), so the nearest close within this window is used instead.
FISCAL_PRICE_TOLERANCE_DAYS = 7


def _cache_path(ticker: str, kind: str) -> Path:
    return CACHE_DIR / f"{ticker.upper().replace('/', '_')}.{kind}.json"


def _read_cache(path: Path, ttl: float) -> dict | None:
    if ttl <= 0 or not path.exists():
        return None
    if time.time() - path.stat().st_mtime > ttl:
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # A truncated cache file is a nuisance, not an error worth aborting a run for:
        # drop it and refetch.
        return None


def _write_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    os.replace(tmp, path)


def _number(value) -> float | None:
    """Coerce a provider value to a float, rejecting NaN and non-numerics.

    NaN matters specifically: pandas hands back float('nan') for missing statement rows,
    and NaN propagates silently through arithmetic to produce a fair value of nan that
    compares False against every threshold and lands in whichever branch is last.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if result != result else result


def _series(frame, row: str) -> list[float]:
    """One statement row as an oldest-first list, dropping years with no data.

    yfinance returns columns newest-first, and the five-year frames routinely carry a
    trailing all-NaN column for a year that was never filed.
    """
    if frame is None or getattr(frame, "empty", True) or row not in frame.index:
        return []
    values = [_number(v) for v in frame.loc[row].tolist()]
    return [v for v in reversed(values) if v is not None]


def _historical_pe(ticker: yf.Ticker, income_statement) -> list[float]:
    """The multiple this stock actually traded at, at each fiscal year end.

    Computed from the close nearest each fiscal period end divided by that year's diluted
    EPS, rather than from any provider-supplied historical ratio: the ratio fields report
    a single point in time and there is no field for 'what it was five years ago'.
    """
    if income_statement is None or getattr(income_statement, "empty", True):
        return []
    if "Diluted EPS" not in income_statement.index:
        return []
    try:
        prices = ticker.history(period="6y", auto_adjust=True)
    except Exception:
        return []
    if prices.empty:
        return []

    closes = prices["Close"]
    tolerance = f"{FISCAL_PRICE_TOLERANCE_DAYS}D"
    multiples = []
    for period_end in income_statement.columns:
        eps = _number(income_statement.loc["Diluted EPS", period_end])
        if not eps or eps <= 0:
            continue
        try:
            stamp = period_end.tz_localize(closes.index.tz) if period_end.tzinfo is None \
                else period_end.tz_convert(closes.index.tz)
            nearest = closes.index.get_indexer([stamp], method="nearest",
                                               tolerance=tolerance)[0]
        except (TypeError, ValueError, KeyError):
            continue
        if nearest == -1:
            continue
        multiples.append(float(closes.iloc[nearest]) / eps)
    return multiples


def fetch(ticker: str, *, quote_ttl: float = QUOTE_TTL_SECONDS,
          statement_ttl: float = STATEMENT_TTL_SECONDS) -> Fundamentals:
    """Everything the models need for one ticker, cached in two tiers.

    Raises ValueError when the ticker has no price at all, which is how a typo or a
    delisting shows up; every other missing field is left as None for the anchors to
    abstain on.
    """
    symbol = ticker.upper().strip()
    quote_file = _cache_path(symbol, "quote")
    statement_file = _cache_path(symbol, "statements")

    handle = None
    quote = _read_cache(quote_file, quote_ttl)
    if quote is None:
        handle = yf.Ticker(symbol)
        info = handle.info or {}
        quote = {k: info.get(k) for k in (
            "currentPrice", "regularMarketPrice", "previousClose", "shortName", "longName",
            "currency", "quoteType", "sector", "trailingEps", "bookValue",
            "freeCashflow", "sharesOutstanding", "totalCash", "totalDebt", "beta",
            "dividendRate", "payoutRatio", "returnOnEquity", "debtToEquity",
            "earningsGrowth", "targetMeanPrice")}
        quote["fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_cache(quote_file, quote)

    statements = _read_cache(statement_file, statement_ttl)
    if statements is None:
        handle = handle or yf.Ticker(symbol)
        try:
            cashflow, income = handle.cashflow, handle.income_stmt
        except Exception:
            cashflow = income = None
        statements = {
            "fcf_history": _series(cashflow, "Free Cash Flow"),
            "eps_history": _series(income, "Diluted EPS"),
            "historical_pe": _historical_pe(handle, income),
        }
        _write_cache(statement_file, statements)

    price = (_number(quote.get("currentPrice")) or _number(quote.get("regularMarketPrice"))
             or _number(quote.get("previousClose")))
    if price is None or price <= 0:
        raise ValueError(f"{symbol}: no price available (delisted or unknown ticker?)")

    # Yahoo's `freeCashflow` summary field disagrees badly with the company's own cash
    # flow statement -- for MSFT it reported 16.5bn against a filed 67bn, which fed a
    # DCF that valued the share at a sixth of its price. The annual statement is the
    # audited number, so it wins; the summary field is only a fallback for companies
    # whose statements did not parse.
    fcf_history = [v for v in statements.get("fcf_history", []) if v is not None]
    free_cash_flow = fcf_history[-1] if fcf_history else _number(quote.get("freeCashflow"))

    dividend_rate = _number(quote.get("dividendRate"))
    return Fundamentals(
        ticker=symbol,
        price=price,
        currency=quote.get("currency") or "USD",
        quote_type=quote.get("quoteType") or "EQUITY",
        name=quote.get("shortName") or quote.get("longName") or symbol,
        sector=quote.get("sector") or "",
        eps_trailing=_number(quote.get("trailingEps")),
        book_value_per_share=_number(quote.get("bookValue")),
        free_cash_flow=free_cash_flow,
        shares_outstanding=_number(quote.get("sharesOutstanding")),
        total_cash=_number(quote.get("totalCash")),
        total_debt=_number(quote.get("totalDebt")),
        beta=_number(quote.get("beta")),
        dividend_rate=dividend_rate,
        # Derived from the rate rather than read from `dividendYield`, whose units have
        # changed between yfinance releases (0.32 meaning 0.32% in some, 32% in others).
        # A 100x error here would silently swing the dividend anchor by two orders.
        dividend_yield=(dividend_rate / price) if dividend_rate else None,
        payout_ratio=_number(quote.get("payoutRatio")),
        return_on_equity=_number(quote.get("returnOnEquity")),
        debt_to_equity=_number(quote.get("debtToEquity")),
        earnings_growth=_number(quote.get("earningsGrowth")),
        analyst_target=_number(quote.get("targetMeanPrice")),
        fcf_history=fcf_history,
        eps_history=[v for v in statements.get("eps_history", []) if v is not None],
        historical_pe=[v for v in statements.get("historical_pe", []) if v is not None],
        fetched_at=quote.get("fetched_at", ""),
    )
