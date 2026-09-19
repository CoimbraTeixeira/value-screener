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

# Share-count change is clamped before it reaches the DCF. A one-off acquisition paid in
# stock, or a buyback funded by a single asset sale, is not a policy that runs for ten
# years, and extrapolating it as one dominates the result.
MAX_SHARE_GROWTH = 0.10
MIN_SHARE_GROWTH = -0.05

# Net downward estimate revisions this severe mark a business whose forecasts are being
# cut. Cheapness plus falling estimates is the shape of a value trap, so this vetoes a
# BUY rather than adjusting a number.
FALLING_ESTIMATES_RATIO = -0.30
FALLING_ESTIMATES_DRIFT = -0.02

# Analyst coverage below this makes the forward estimates one or two people's opinion.
MIN_ANALYST_COVERAGE = 3

# A results release can move a share more than a quarter of screening does, so a buy
# decision inside this window is a coin toss on the print.
EARNINGS_SOON_DAYS = 7

# Insider selling below this share of the company is routine compensation mechanics, not
# a position change worth reading anything into.
MATERIAL_INSIDER_SALE = 0.005

# Sectors where free cash flow and balance-sheet leverage do not mean what they mean
# elsewhere. A bank's lending is an investing outflow and its deposits are liabilities;
# a REIT's whole business is buying buildings with mortgages. Both report negative FCF
# and high leverage while entirely healthy, so those two gates are suspended here rather
# than condemning the sectors wholesale.
FCF_EXEMPT_SECTORS = frozenset({"Financial Services", "Real Estate"})


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
    # Plain-language description of the business. Carried for the semantic index, which
    # embeds what a company does; no valuation model reads it.
    business_summary: str = ""

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
    # Forward-looking inputs, absent unless the caller fetched them.
    forward: "Forward | None" = None


@dataclass
class Forward:
    """Analyst estimates and share-count trend: what the market expects next.

    Split from Fundamentals because these are a different kind of input. Everything
    above is reported history; everything here is a forecast, and forecasts are wrong in
    a direction -- sell-side estimates are persistently optimistic, most of all for the
    companies in most trouble. So these widen or veto a conclusion far more often than
    they raise a fair value.

    Analyst *price targets* are deliberately not among the valuation inputs. They track
    the current price with a lag and adding them would launder consensus into an
    estimate whose whole purpose is to disagree with consensus. They are carried only to
    be displayed as a contrast.
    """

    eps_next_year: float | None = None
    eps_year_after: float | None = None
    revenue_growth_next_year: float | None = None
    long_term_growth: float | None = None
    analyst_count: int | None = None

    # Estimate momentum: how many analysts moved which way in the last 30 days, and how
    # far the consensus for next year has travelled in 90.
    revisions_up: int | None = None
    revisions_down: int | None = None
    eps_drift_90d: float | None = None

    target_high: float | None = None
    target_low: float | None = None

    # Annualised change in share count. Positive is dilution, negative is buyback.
    share_growth: float | None = None

    days_to_earnings: int | None = None
    insider_net_shares_6m: float | None = None


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

    Takes the *lowest* of realised free-cash-flow growth, the analyst earnings growth
    estimate, the long-term growth forecast and next year's revenue growth estimate.
    They disagree often, and when they do the conservative one is the one least
    dependent on a forecast being right -- sell-side estimates are persistently
    optimistic and most so for companies in trouble, which is exactly when a screener
    must not be. Absent all of them, growth is zero: a no-growth DCF still values the
    existing cash stream, which is a defensible floor.
    """
    candidates = [cagr(fundamentals.fcf_history), fundamentals.earnings_growth]
    forward = fundamentals.forward
    if forward:
        candidates += [forward.long_term_growth, forward.revenue_growth_next_year]
    usable = [g for g in candidates if g is not None]
    if not usable:
        return MIN_STAGE1_GROWTH
    return min(MAX_STAGE1_GROWTH, max(MIN_STAGE1_GROWTH, min(usable)))


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

    # Future cash flows are split across future shares, so a company issuing stock is
    # worth less per share even with identical cash flows. Dividing each year's flow by
    # the grown share count is exact under constant issuance, and avoids the usual fudge
    # of valuing tomorrow's cash against today's share count. Buybacks run the same
    # arithmetic in reverse.
    share_growth = 0.0
    if f.forward and f.forward.share_growth is not None:
        share_growth = min(MAX_SHARE_GROWTH, max(MIN_SHARE_GROWTH, f.forward.share_growth))

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
        dilution = (1.0 + share_growth) ** year
        present_value += cash_flow / dilution / (1.0 + rate) ** year

    terminal_value = cash_flow * (1.0 + terminal_growth) / (rate - terminal_growth)
    present_value += (terminal_value / (1.0 + share_growth) ** DCF_YEARS
                      / (1.0 + rate) ** DCF_YEARS)

    # Net cash belongs to today's shareholders, so it is not diluted by future issuance.
    net_cash = (f.total_cash or 0.0) - (f.total_debt or 0.0)
    per_share = (present_value + net_cash) / f.shares_outstanding
    if per_share <= 0:
        # Net debt can exceed the discounted cash stream. That is a real result, but it
        # is a solvency statement rather than a price, so it is not offered as a target.
        return Anchor("dcf", None, "net debt exceeds discounted cash flows")

    detail = f"{growth:.1%} growth, {rate:.1%} discount, {terminal_growth:.1%} terminal"
    if share_growth:
        detail += (f", {abs(share_growth):.1%}/yr "
                   f"{'dilution' if share_growth > 0 else 'buyback'}")
    return Anchor("dcf", per_share, detail)


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


def forward_pe_anchor(f: Fundamentals, max_pe: float = DEFAULT_MAX_ANCHOR_PE,
                      risk_free: float = DEFAULT_RISK_FREE,
                      equity_premium: float = DEFAULT_EQUITY_PREMIUM) -> Anchor:
    """Next year's consensus earnings at this stock's own historical multiple.

    The one place analyst forecasts earn a vote. It is the only anchor that can see a
    recovery coming -- every other model here reads the past, so a company emerging from
    a bad year is permanently condemned by trailing EPS. The estimate is discounted back
    one year, because a value that arrives twelve months from now is not worth its face
    amount today.

    Guarded by analyst coverage: with one or two estimates this is an opinion, not a
    consensus, and it would otherwise carry the same weight as four years of filings.
    """
    forward = f.forward
    if not forward or not forward.eps_next_year or forward.eps_next_year <= 0:
        return Anchor("fwd_pe", None, "no positive forward EPS estimate")
    if (forward.analyst_count or 0) < MIN_ANALYST_COVERAGE:
        return Anchor("fwd_pe", None,
                      f"only {forward.analyst_count or 0} analysts covering")
    usable = [pe for pe in f.historical_pe if pe and pe > 0]
    if len(usable) < 2:
        return Anchor("fwd_pe", None, "insufficient multiple history")

    typical = median(usable)
    note = f"{forward.eps_next_year:,.2f} est. EPS at {typical:.1f}x"
    if typical > max_pe:
        typical = max_pe
        note = f"{forward.eps_next_year:,.2f} est. EPS at {max_pe:.0f}x (capped)"
    rate = discount_rate(f.beta, risk_free, equity_premium)
    return Anchor("fwd_pe", typical * forward.eps_next_year / (1.0 + rate),
                  note + f", discounted a year at {rate:.1%}")


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

# --- Plain buy / hold / sell -----------------------------------------------------------
#
# The six verdicts above describe *why* a stock landed where it did, which is what makes
# them worth reading and also what makes them a poor notification. These collapse them
# into the three words a morning message can carry.
#
# One honest limit is baked into the wording. Without a cost basis -- a Yahoo watchlist
# export has none -- SELL cannot mean "close your position", because nothing here knows
# you hold one, what you paid, or what the tax would be. It means "this is not worth
# owning at this price". For real holdings with quantity and cost, portfolio.review()
# answers the stronger question with EXIT/TRIM/HOLD/ADD.

RECOMMEND_BUY = "BUY"
RECOMMEND_HOLD = "HOLD"
RECOMMEND_SELL = "SELL"
RECOMMEND_NONE = "N/A"

# How far above fair value a stock must trade before the recommendation turns from
# "do not add" into "do not own". Matches portfolio.TRIM_PREMIUM deliberately: the same
# gap should not mean TRIM in one table and HOLD in another.
SELL_PREMIUM = 0.25


def recommendation(assessment: "Assessment") -> tuple[str, str]:
    """One of BUY / HOLD / SELL / N/A, with the reason in a few words.

    A quality-gate failure is a SELL regardless of price: the gates fire on unprofitable
    or cash-burning businesses, and a cheap price is not a reason to own one. Mild
    overvaluation is HOLD rather than SELL, because the estimate's own error bar is
    wider than ten percent and churning on that noise costs spread for nothing.
    """
    if assessment.verdict == NO_DATA:
        return RECOMMEND_NONE, "these models do not apply"
    if assessment.verdict == AVOID:
        # One failure plus a count, not the whole list: a business failing four gates is
        # not four times as informative as one failing a single gate, and the full text
        # runs past 150 characters on the worst names.
        first = assessment.gate_failures[0] if assessment.gate_failures else "failed quality gates"
        extra = len(assessment.gate_failures) - 1
        return RECOMMEND_SELL, first + (f" (+{extra} more)" if extra > 0 else "")
    if assessment.verdict == BUY:
        return RECOMMEND_BUY, f"{assessment.margin_of_safety:.0%} below estimate"

    margin = assessment.margin_of_safety
    if margin is None:
        return RECOMMEND_NONE, "no usable estimate"

    # Margin is (fair - price) / fair, so a price 25% above fair value is a margin of
    # -0.25/1.25 = -20%. Comparing in price terms keeps this aligned with SELL_PREMIUM.
    if assessment.fair_value and assessment.price > assessment.fair_value * (1 + SELL_PREMIUM):
        premium = assessment.price / assessment.fair_value - 1
        return RECOMMEND_SELL, f"{premium:.0%} above estimate"

    if assessment.verdict == WATCH and margin >= WATCH_MARGIN:
        # Cheap enough to be interesting but blocked by disagreement or falling
        # estimates; the note says which.
        reason = assessment.notes[0] if assessment.notes else (
            assessment.flags[0] if assessment.flags else f"{margin:.0%} below estimate")
        return RECOMMEND_HOLD, reason
    return RECOMMEND_HOLD, f"{margin:+.0%} vs estimate"


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
    # Forward-looking warnings. Distinct from gate_failures: a gate says the business is
    # broken today, a flag says the estimate ahead of it is not to be leaned on.
    flags: list[str] = field(default_factory=list)
    # Carried through from the fundamentals purely so the semantic index can embed the
    # description alongside the verdict without refetching.
    sector: str = ""
    business_summary: str = ""

    @property
    def usable_anchors(self) -> list[Anchor]:
        return [a for a in self.anchors if a.value is not None]


def check_gates(f: Fundamentals, max_debt_to_equity: float = MAX_DEBT_TO_EQUITY,
                min_roe: float = MIN_RETURN_ON_EQUITY) -> list[str]:
    """Business-quality failures, in plain language. Empty list means nothing disqualifying.

    A missing field is not treated as a failure. Absent data is a gap in the feed, not
    evidence of a bad business, and failing on it would quietly blacklist every company
    whose filings the provider parses poorly.

    Two gates are suspended by sector rather than applied everywhere. Free cash flow is
    not a meaningful concept for a bank, whose lending shows up as investing outflow, or
    for a REIT, whose entire business is capital expenditure -- Equinix builds data
    centres and so reports negative FCF every year of a perfectly sound decade. Balance
    sheet leverage is likewise normal for both: a bank's deposits and a REIT's mortgages
    are the business model, not distress. Applying either gate there produces confident
    false negatives on whole sectors.
    """
    failures = []
    cash_flow_exempt = f.sector in FCF_EXEMPT_SECTORS

    if f.eps_trailing is not None and f.eps_trailing <= 0:
        failures.append("unprofitable (negative trailing EPS)")
    if (not cash_flow_exempt and f.free_cash_flow is not None
            and f.free_cash_flow <= 0):
        failures.append("burning cash (negative free cash flow)")
    if (not cash_flow_exempt and f.debt_to_equity is not None
            and f.debt_to_equity > max_debt_to_equity):
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


def estimates_falling(forward: Forward | None) -> bool:
    """Whether next year's consensus is being cut.

    Two independent readings, either of which counts: the balance of analysts revising
    down over the last 30 days, and how far the consensus itself has drifted over 90.
    The count catches a sharp recent turn; the drift catches a slow grind that never
    shows up as a dramatic week.
    """
    if forward is None:
        return False
    up, down = forward.revisions_up, forward.revisions_down
    if up is not None and down is not None and (up + down) > 0:
        if (up - down) / (up + down) <= FALLING_ESTIMATES_RATIO:
            return True
    drift = forward.eps_drift_90d
    return drift is not None and drift <= FALLING_ESTIMATES_DRIFT


def forward_flags(f: Fundamentals) -> list[str]:
    """Forward-looking warnings, in plain language.

    None of these touch the fair value. They describe how much weight the estimate will
    bear, which is a separate question from what the number is, and folding them into
    the price would produce a single figure that quietly encodes four opinions.
    """
    forward = f.forward
    if forward is None:
        return []

    flags = []
    if estimates_falling(forward):
        up, down = forward.revisions_up or 0, forward.revisions_down or 0
        drift = forward.eps_drift_90d
        detail = f"{down} down vs {up} up in 30d" if (up + down) else ""
        if drift is not None and drift <= FALLING_ESTIMATES_DRIFT:
            detail = (detail + ", " if detail else "") + f"consensus {drift:.1%} in 90d"
        flags.append(f"estimates falling ({detail})")

    if forward.share_growth is not None and forward.share_growth > 0.02:
        flags.append(f"diluting {forward.share_growth:.1%}/yr")

    if (forward.analyst_count or 0) and forward.analyst_count < MIN_ANALYST_COVERAGE:
        flags.append(f"thin coverage ({forward.analyst_count} analysts)")

    if forward.days_to_earnings is not None and 0 <= forward.days_to_earnings <= EARNINGS_SOON_DAYS:
        flags.append(f"reports in {forward.days_to_earnings}d")

    if forward.target_high and forward.target_low and f.price > 0:
        span = (forward.target_high - forward.target_low) / f.price
        if span > 0.8:
            flags.append(f"analysts span {span:.0%} of price")

    # Insider *selling* is close to universal -- option exercises and scheduled 10b5-1
    # plans generate it continuously at healthy companies, so flagging it fires on
    # almost every stock and carries no information. Only buying, which an insider has
    # no routine reason to do, and selling large enough to be a real change of
    # position, are worth a line.
    net_insider = forward.insider_net_shares_6m
    if net_insider and f.shares_outstanding:
        share = abs(net_insider) / f.shares_outstanding
        if net_insider > 0:
            flags.append(f"insiders net buyers ({share:.2%} of shares)")
        elif share >= MATERIAL_INSIDER_SALE:
            flags.append(f"heavy insider selling ({share:.2%} of shares)")

    return flags


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
        forward_pe_anchor(f, max_anchor_pe, risk_free, equity_premium),
        graham_anchor(f),
        dividend_anchor(f, risk_free, equity_premium),
    ]
    gate_failures = check_gates(f, max_debt_to_equity, min_roe)
    flags = forward_flags(f)
    notes: list[str] = []

    if f.quote_type and f.quote_type != "EQUITY":
        # A fund has no earnings, book value or cash flow of its own; every anchor here
        # would be measuring its holdings by accident.
        return Assessment(f.ticker, f.name, f.price, f.currency, NO_DATA, None, None,
                          anchors, gate_failures,
                          [f"{f.quote_type} is not a single company; these models do not apply"],
                          f.analyst_target, flags, f.sector, f.business_summary)

    usable = [a for a in anchors if a.value is not None]
    if len(usable) < 2:
        # A company can fail the quality gates precisely *because* it is unprofitable and
        # cash-burning -- the same facts that leave the anchors nothing to work with.
        # Reporting that as "no data" would hide a conclusion already in hand.
        return Assessment(f.ticker, f.name, f.price, f.currency,
                          AVOID if gate_failures else NO_DATA, None, None,
                          anchors, gate_failures,
                          ["fewer than two anchors could be computed"], f.analyst_target,
                          flags, f.sector, f.business_summary)

    fair_value = median(a.value for a in usable)
    margin = (fair_value - f.price) / fair_value if fair_value > 0 else None

    spread = dispersion(usable)
    wide = spread is not None and spread > MAX_TRUSTED_DISPERSION
    if wide:
        notes.append(f"anchors disagree {spread:.1f}x -- estimate is weak")

    falling = estimates_falling(f.forward)

    if gate_failures:
        verdict = AVOID
    elif margin is None:
        verdict = NO_DATA
    elif margin >= buy_margin:
        # Two ways a large discount fails to be a finding. Anchors that contradict each
        # other are noise with a decimal point. Cheapness while the forecasts underneath
        # it are being cut is the shape of a value trap: the price has fallen because the
        # earnings are going to, and every backward-looking anchor is still pricing the
        # earnings that are about to disappear.
        verdict = WATCH if (wide or falling) else BUY
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
                      anchors, gate_failures, notes, f.analyst_target, flags,
                      f.sector, f.business_summary)
