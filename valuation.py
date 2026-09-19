"""Valuation maths, deliberately kept free of network and I/O so every number here is
reproducible from a fixture.

The design assumption is that no single model is trustworthy. Each anchor below is a
different, independently-wrong estimate of what a share is worth; what matters is
whether they *agree*. Four anchors clustered inside a narrow band is a weak signal
worth acting on. Four anchors spread over a 4x range means the business is not
valuable by any method this file can apply, and the screener says so rather than
averaging the disagreement away into a confident-looking number.

Anchors abstain (return None with a reason) rather than guessing. A company with
negative earnings has no meaningful P/E anchor, and inventing one by using an
absolute-value EPS or a placeholder multiple would produce a fair value that looks
like the others and is pure noise.
"""

from dataclasses import dataclass, field
from statistics import median

# CAPM inputs. These are assumptions, not facts, which is why they are surfaced as CLI
# flags rather than buried: the discount rate is the single input a DCF is most
# sensitive to, and a user who disagrees with 4%/5% should be able to say so.
DEFAULT_RISK_FREE = 0.04
DEFAULT_EQUITY_PREMIUM = 0.05

# A discount rate outside this band stops describing a listed equity. Beta is noisy and
# occasionally absurd (a thin-float stock can print beta 4), and an unclamped CAPM rate
# then drives the DCF to a near-zero or wildly inflated value purely on that artefact.
MIN_DISCOUNT = 0.07
MAX_DISCOUNT = 0.15

# Long-run nominal growth. A terminal rate above this says the company eventually
# becomes the whole economy, which no business has ever done.
DEFAULT_TERMINAL_GROWTH = 0.025

# Stage-one growth is capped because extrapolating a recent high growth rate for five
# years is the most common way a DCF produces a flattering answer. A company really
# compounding faster than this will simply be understated, which is the safe direction.
MAX_STAGE1_GROWTH = 0.12
MIN_STAGE1_GROWTH = 0.0

DCF_YEARS = 10
DCF_STAGE1_YEARS = 5

# Graham's original formula used 22.5 = 15x earnings * 1.5x book. It is retained at its
# original value because the number is the model; tuning it upward to make modern
# software companies pass would just be reinventing the P/E anchor with extra steps.
GRAHAM_CONSTANT = 22.5

# The historical multiple a stock traded at is only an anchor if it was ever sane. Above
# this the anchor is reporting a past bubble, not a normal valuation, so it is capped and
# the cap is reported rather than applied silently.
DEFAULT_MAX_ANCHOR_PE = 30.0

# Below this yield a dividend is a rounding error on total return and the dividend
# discount model says almost nothing about the share's worth.
MIN_MEANINGFUL_YIELD = 0.015

# Ratio of highest to lowest anchor above which the anchors are judged to disagree. Two
# anchors a third apart are the same answer with different assumptions; a 2.5x spread
# means at least one model does not fit this business.
MAX_TRUSTED_DISPERSION = 2.5


@dataclass
class Fundamentals:
    """The inputs every anchor draws on, already normalised.

    Everything is Optional because a real feed has holes: newly-listed companies have no
    multi-year history, loss-making ones have no EPS, and some fields are simply absent
    for non-US listings. Anchors are written to abstain on missing inputs.
    """

    ticker: str
    price: float
    currency: str = "USD"
    quote_type: str = "EQUITY"
    name: str = ""
    sector: str = ""

    eps_trailing: float | None = None
    book_value_per_share: float | None = None
    free_cash_flow: float | None = None
    shares_outstanding: float | None = None
    total_cash: float | None = None
    total_debt: float | None = None
    beta: float | None = None
    dividend_rate: float | None = None
    dividend_yield: float | None = None      # fraction, e.g. 0.032 for 3.2%
    payout_ratio: float | None = None
    return_on_equity: float | None = None
    debt_to_equity: float | None = None      # percent, as reported (78.4 = 0.78x)
    earnings_growth: float | None = None
    analyst_target: float | None = None

    # Oldest-first annual series used for growth and historical multiples.
    fcf_history: list[float] = field(default_factory=list)
    eps_history: list[float] = field(default_factory=list)
    historical_pe: list[float] = field(default_factory=list)

    fetched_at: str = ""


@dataclass
class Anchor:
    """One model's answer, or its refusal to give one."""

    name: str
    value: float | None
    detail: str


def cagr(series: list[float], years: int | None = None) -> float | None:
    """Compound annual growth between the first and last point.

    Returns None when the endpoints cannot express growth: a run that starts at or below
    zero has no defined rate, and reporting one (or silently flipping the sign) would
    feed a fictional growth rate straight into the DCF.
    """
    clean = [v for v in series if v is not None]
    if len(clean) < 2:
        return None
    start, end = clean[0], clean[-1]
    if start <= 0 or end <= 0:
        return None
    periods = years if years is not None else len(clean) - 1
    if periods <= 0:
        return None
    return (end / start) ** (1.0 / periods) - 1.0


def discount_rate(beta: float | None, risk_free: float = DEFAULT_RISK_FREE,
                  equity_premium: float = DEFAULT_EQUITY_PREMIUM) -> float:
    """CAPM cost of equity, clamped to a range that still describes a stock."""
    effective_beta = 1.0 if beta is None or beta <= 0 else beta
    return min(MAX_DISCOUNT, max(MIN_DISCOUNT, risk_free + effective_beta * equity_premium))


def estimate_growth(fundamentals: Fundamentals) -> float:
    """Stage-one growth rate for the DCF.

    Takes the *lower* of realised free-cash-flow growth and the analyst earnings growth
    estimate. The two disagree often, and when they do the conservative one is the one
    that does not depend on a forecast being right. Absent both, growth is zero: a
    no-growth DCF still values the existing cash stream, which is a defensible floor.
    """
    candidates = [g for g in (cagr(fundamentals.fcf_history), fundamentals.earnings_growth)
                  if g is not None]
    if not candidates:
        return MIN_STAGE1_GROWTH
    return min(MAX_STAGE1_GROWTH, max(MIN_STAGE1_GROWTH, min(candidates)))


def dcf_anchor(f: Fundamentals, risk_free: float = DEFAULT_RISK_FREE,
               equity_premium: float = DEFAULT_EQUITY_PREMIUM,
               terminal_growth: float = DEFAULT_TERMINAL_GROWTH) -> Anchor:
    """Two-stage discounted free cash flow, plus net cash, per share.

    Stage one grows at the estimated rate for five years; stage two fades that rate
    linearly to the terminal rate over the next five, which avoids the cliff-edge
    discontinuity of a single-stage model where year 11 abruptly drops from 12% to 2.5%
    growth and the terminal value inherits an implausible base.
    """
    if not f.free_cash_flow or f.free_cash_flow <= 0:
        return Anchor("dcf", None, "no positive free cash flow")
    if not f.shares_outstanding or f.shares_outstanding <= 0:
        return Anchor("dcf", None, "share count unavailable")

    rate = discount_rate(f.beta, risk_free, equity_premium)
    if terminal_growth >= rate:
        # Gordon growth diverges here and would return a negative or infinite value.
        return Anchor("dcf", None, f"terminal growth {terminal_growth:.1%} exceeds "
                                   f"discount rate {rate:.1%}")

    growth = estimate_growth(f)
    cash_flow = f.free_cash_flow
    present_value = 0.0
    fade_years = DCF_YEARS - DCF_STAGE1_YEARS
    for year in range(1, DCF_YEARS + 1):
        if year <= DCF_STAGE1_YEARS:
            year_growth = growth
        else:
            progress = (year - DCF_STAGE1_YEARS) / fade_years
            year_growth = growth + (terminal_growth - growth) * progress
        cash_flow *= 1.0 + year_growth
        present_value += cash_flow / (1.0 + rate) ** year

    terminal_value = cash_flow * (1.0 + terminal_growth) / (rate - terminal_growth)
    present_value += terminal_value / (1.0 + rate) ** DCF_YEARS

    net_cash = (f.total_cash or 0.0) - (f.total_debt or 0.0)
    per_share = (present_value + net_cash) / f.shares_outstanding
    if per_share <= 0:
        # Net debt can exceed the discounted cash stream. That is a real result, but it
        # is a solvency statement rather than a price, so it is not offered as a target.
        return Anchor("dcf", None, "net debt exceeds discounted cash flows")
    return Anchor("dcf", per_share,
                  f"{growth:.1%} growth, {rate:.1%} discount, {terminal_growth:.1%} terminal")


def historical_pe_anchor(f: Fundamentals, max_pe: float = DEFAULT_MAX_ANCHOR_PE) -> Anchor:
    """Current earnings valued at the multiple this stock has historically commanded.

    Uses the company's own past multiple rather than a sector average, because a sector
    median silently assumes the business deserves to be valued like its peers -- which is
    the entire question being asked. Mean-reversion to a stock's own multiple is a
    narrower and more defensible claim.
    """
    if not f.eps_trailing or f.eps_trailing <= 0:
        return Anchor("hist_pe", None, "no positive trailing EPS")
    usable = [pe for pe in f.historical_pe if pe and pe > 0]
    if len(usable) < 2:
        return Anchor("hist_pe", None, "insufficient multiple history")

    typical = median(usable)
    note = f"median {typical:.1f}x over {len(usable)}y"
    if typical > max_pe:
        typical = max_pe
        note += f", capped at {max_pe:.0f}x"
    return Anchor("hist_pe", typical * f.eps_trailing, note)


def graham_anchor(f: Fundamentals) -> Anchor:
    """Graham's number: sqrt(22.5 * EPS * book value per share).

    Deeply conservative and openly hostile to asset-light businesses, which is why it is
    one vote rather than the answer. When it is the only anchor far below the others,
    that usually means the company's value is in intangibles the balance sheet omits --
    information worth seeing rather than smoothing over.
    """
    if not f.eps_trailing or f.eps_trailing <= 0:
        return Anchor("graham", None, "no positive trailing EPS")
    if not f.book_value_per_share or f.book_value_per_share <= 0:
        return Anchor("graham", None, "no positive book value")
    return Anchor("graham", (GRAHAM_CONSTANT * f.eps_trailing * f.book_value_per_share) ** 0.5,
                  "sqrt(22.5 x EPS x book)")


def dividend_anchor(f: Fundamentals, risk_free: float = DEFAULT_RISK_FREE,
                    equity_premium: float = DEFAULT_EQUITY_PREMIUM) -> Anchor:
    """Gordon growth on the dividend, for shares actually held for income.

    Dividend growth is taken as the retained-earnings growth rate (ROE x retention),
    which ties the payout's future to the business's own economics instead of
    extrapolating the last few raises.
    """
    if not f.dividend_rate or f.dividend_rate <= 0:
        return Anchor("dividend", None, "pays no dividend")
    if (f.dividend_yield or 0.0) < MIN_MEANINGFUL_YIELD:
        return Anchor("dividend", None,
                      f"yield below {MIN_MEANINGFUL_YIELD:.1%}, immaterial to value")

    rate = discount_rate(f.beta, risk_free, equity_premium)
    retention = 1.0 - min(1.0, max(0.0, f.payout_ratio or 0.0))
    growth = min(DEFAULT_TERMINAL_GROWTH + 0.02, max(0.0, (f.return_on_equity or 0.0) * retention))
    if growth >= rate:
        return Anchor("dividend", None, "implied dividend growth exceeds discount rate")
    return Anchor("dividend", f.dividend_rate * (1.0 + growth) / (rate - growth),
                  f"{growth:.1%} dividend growth, {rate:.1%} discount")


# --- Quality gates -------------------------------------------------------------------
#
# Price is the second question. A cheap share in a business that loses money, burns cash
# or is drowning in debt is usually cheap for a reason, and every anchor above will
# happily price it anyway. These gates are the veto: fail one and the verdict is AVOID
# no matter how large the apparent discount.

MAX_DEBT_TO_EQUITY = 200.0   # percent, as reported: 200 = 2.0x equity
MIN_RETURN_ON_EQUITY = 0.0

# Margin of safety bands. 30% is the conventional Graham threshold and exists to absorb
# the error in the fair value estimate, not to predict a 30% gain.
BUY_MARGIN = 0.30
WATCH_MARGIN = 0.10
FAIR_MARGIN = -0.10

BUY = "BUY"
WATCH = "WATCH"
FAIR = "FAIR"
EXPENSIVE = "EXPENSIVE"
AVOID = "AVOID"
NO_DATA = "NO DATA"


@dataclass
class Assessment:
    ticker: str
    name: str
    price: float
    currency: str
    verdict: str
    fair_value: float | None
    margin_of_safety: float | None
    anchors: list[Anchor]
    gate_failures: list[str]
    notes: list[str]
    analyst_target: float | None = None

    @property
    def usable_anchors(self) -> list[Anchor]:
        return [a for a in self.anchors if a.value is not None]


def check_gates(f: Fundamentals, max_debt_to_equity: float = MAX_DEBT_TO_EQUITY,
                min_roe: float = MIN_RETURN_ON_EQUITY) -> list[str]:
    """Business-quality failures, in plain language. Empty list means nothing disqualifying.

    A missing field is not treated as a failure. Absent data is a gap in the feed, not
    evidence of a bad business, and failing on it would quietly blacklist every company
    whose filings the provider parses poorly.
    """
    failures = []
    if f.eps_trailing is not None and f.eps_trailing <= 0:
        failures.append("unprofitable (negative trailing EPS)")
    if f.free_cash_flow is not None and f.free_cash_flow <= 0:
        failures.append("burning cash (negative free cash flow)")
    if f.debt_to_equity is not None and f.debt_to_equity > max_debt_to_equity:
        failures.append(f"leverage {f.debt_to_equity / 100:.1f}x equity "
                        f"(limit {max_debt_to_equity / 100:.1f}x)")
    if f.return_on_equity is not None and f.return_on_equity < min_roe:
        failures.append(f"return on equity {f.return_on_equity:.1%}")
    return failures


def dispersion(anchors: list[Anchor]) -> float | None:
    """Highest anchor divided by lowest: how much the models disagree."""
    values = [a.value for a in anchors if a.value is not None and a.value > 0]
    if len(values) < 2:
        return None
    return max(values) / min(values)


def assess(f: Fundamentals, *, risk_free: float = DEFAULT_RISK_FREE,
           equity_premium: float = DEFAULT_EQUITY_PREMIUM,
           terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
           max_anchor_pe: float = DEFAULT_MAX_ANCHOR_PE,
           max_debt_to_equity: float = MAX_DEBT_TO_EQUITY,
           min_roe: float = MIN_RETURN_ON_EQUITY,
           buy_margin: float = BUY_MARGIN) -> Assessment:
    """Run every anchor, take the median, and grade the discount against it.

    The median rather than the mean: one model blowing up (a DCF on a company whose cash
    flow just spiked) should not drag the estimate with it, and with three or four
    anchors the median is the one that survives a single bad input.
    """
    anchors = [
        dcf_anchor(f, risk_free, equity_premium, terminal_growth),
        historical_pe_anchor(f, max_anchor_pe),
        graham_anchor(f),
        dividend_anchor(f, risk_free, equity_premium),
    ]
    gate_failures = check_gates(f, max_debt_to_equity, min_roe)
    notes: list[str] = []

    if f.quote_type and f.quote_type != "EQUITY":
        # A fund has no earnings, book value or cash flow of its own; every anchor here
        # would be measuring its holdings by accident.
        return Assessment(f.ticker, f.name, f.price, f.currency, NO_DATA, None, None,
                          anchors, gate_failures,
                          [f"{f.quote_type} is not a single company; these models do not apply"],
                          f.analyst_target)

    usable = [a for a in anchors if a.value is not None]
    if len(usable) < 2:
        # A company can fail the quality gates precisely *because* it is unprofitable and
        # cash-burning -- the same facts that leave the anchors nothing to work with.
        # Reporting that as "no data" would hide a conclusion already in hand.
        return Assessment(f.ticker, f.name, f.price, f.currency,
                          AVOID if gate_failures else NO_DATA, None, None,
                          anchors, gate_failures,
                          ["fewer than two anchors could be computed"], f.analyst_target)

    fair_value = median(a.value for a in usable)
    margin = (fair_value - f.price) / fair_value if fair_value > 0 else None

    spread = dispersion(usable)
    wide = spread is not None and spread > MAX_TRUSTED_DISPERSION
    if wide:
        notes.append(f"anchors disagree {spread:.1f}x -- estimate is weak")

    if gate_failures:
        verdict = AVOID
    elif margin is None:
        verdict = NO_DATA
    elif margin >= buy_margin:
        # A large discount computed from anchors that contradict each other is not a
        # finding, it is noise with a decimal point. Downgrade rather than act on it.
        verdict = WATCH if wide else BUY
    elif margin >= WATCH_MARGIN:
        verdict = WATCH
    elif margin >= FAIR_MARGIN:
        verdict = FAIR
    else:
        verdict = EXPENSIVE

    if f.analyst_target and fair_value > 0:
        gap = f.analyst_target / fair_value
        if gap > 1.5 or gap < 0.67:
            notes.append(f"analyst target {f.analyst_target:,.2f} is {gap:.1f}x this estimate")

    return Assessment(f.ticker, f.name, f.price, f.currency, verdict, fair_value, margin,
                      anchors, gate_failures, notes, f.analyst_target)
