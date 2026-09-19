# value-screener

Screens a watchlist for a margin of safety against an estimated fair value.

**This does not know the correct price of anything.** No tool does. What it does is
apply four independent valuation models to each stock, refuse to act when they
disagree, veto on business quality before considering price at all, and show its
working. A `BUY` here means *"passed a mechanical filter, worth reading the filings
for"* — nothing more.

## Usage

```sh
pip install -r requirements.txt
cp watchlist.example.json watchlist.json   # your list; gitignored

./value_screener.py                      # screen watchlist.json
./value_screener.py AAPL MSFT KO         # screen specific tickers
./value_screener.py --explain KO         # every anchor, including abstentions
./value_screener.py --only BUY,WATCH     # just the actionable rows
./value_screener.py --json               # machine-readable
./value_screener.py --notify             # post a summary to Discord
```

Example:

```
   TICKER      PRICE   FAIR EST  MARGIN  VERDICT   NOTE
 x JPM        349.67     255.08    -37%  AVOID     burning cash (negative free cash flow)
 - CSCO       109.51      40.02   -174%  EXPENSIVE analyst target 137.62 is 3.4x this estimate
   MO          69.52      63.43    -10%  FAIR
++ T           25.40      39.74     36%  BUY
```

## How a verdict is reached

**1. Five anchors, each free to abstain.** An anchor returns no value rather than a
guess — a loss-making company has no meaningful P/E, and inventing one produces a
number that looks like the others and is pure noise.

| Anchor | Method | Blind spot |
|---|---|---|
| `dcf` | Two-stage discounted free cash flow + net cash, adjusted for share issuance/buybacks. | Very sensitive to the discount rate. Abstains for banks, which have no meaningful FCF. |
| `hist_pe` | Current EPS × the median multiple *this stock* has traded at. | Anchors to the past. Capped at 30x so a bubble year is not treated as normal. |
| `fwd_pe` | Next year's consensus EPS at that same multiple, discounted back a year. | A forecast. Abstains below 3 analysts. |
| `graham` | `sqrt(22.5 × EPS × book value)` | Hostile to asset-light businesses. Often the low outlier for software. |
| `dividend` | Gordon growth, with dividend growth from ROE × retention. | Abstains below a 1.5% yield. |

`fwd_pe` is the only anchor that can see a recovery coming — every other model reads
the past, so a company emerging from a bad year is permanently condemned by trailing
EPS. Gilead screens at 111 on history alone and 178 with estimates included, because
its trailing loss is a one-off impairment. (The quality gates still say `AVOID`; see
below.)

**2. Quality gates veto first.** Negative EPS, negative free cash flow, leverage above
2.0x equity, or negative ROE ⇒ `AVOID` regardless of how large the discount is. A cheap
share in a failing business is usually cheap for a reason. A *missing* field is not a
failure — that is a gap in the feed, not evidence of a bad business.

**3. Fair value is the median of the surviving anchors.** Median, not mean: one model
blowing up should not drag the estimate.

**4. Disagreement downgrades.** If the highest anchor is more than 2.5x the lowest, a
`BUY` becomes a `WATCH`. A 40% discount computed from models that contradict each other
is noise with a decimal point.

**5. Falling estimates veto a BUY.** Cheap *and* being revised down is the shape of a
value trap: the price fell because the earnings are about to, and every backward-looking
anchor is still pricing earnings that are disappearing. Triggered by the balance of
analyst revisions over 30 days or consensus drift over 90.

| Verdict | Meaning |
|---|---|
| `BUY` | ≥30% below fair value, anchors agree, estimates not falling, gates passed |
| `WATCH` | ≥10% below, or a big discount undermined by disagreement or estimate cuts |
| `FAIR` | Within ±10% |
| `EXPENSIVE` | >10% above |
| `AVOID` | Failed a quality gate |
| `NO DATA` | A fund, or fewer than two anchors computable |

## Forward-looking data

Forward data splits in two, and mixing the halves is how a screener becomes
overconfident. Some of it changes *what a share is worth*; the rest changes *how much
the estimate can be leaned on*. Only the first kind touches the number.

**Priced in** — `fwd_pe` from consensus EPS; the DCF's growth rate, where forecasts join
a `min()` with realised history and so can only ever *lower* it; and share issuance or
buybacks, which divide future cash flows across future shares. That last one is
arithmetic rather than prediction, and is the most reliable addition here.

**Flagged only** — estimate revisions, days to the next results, analyst target spread,
thin coverage, and insider buying. These appear as `FLAG` lines under `--explain`.

**Deliberately excluded** — analyst *price targets*. They track the current price with a
lag, so admitting them would launder consensus into an estimate whose entire purpose is
to disagree with consensus. They are printed beside the result as a contrast and never
enter it.

Insider *selling* is only flagged above 0.5% of shares outstanding. Option exercises and
scheduled 10b5-1 plans make routine selling near-universal — flagged naively it fires on
almost every stock and tells you nothing.

`--no-forward` values on reported history alone, which is a useful A/B: on NKE it is the
difference between `EXPENSIVE` and `FAIR`.

## Using your Yahoo Finance list

Yahoo has no public portfolio API, and the private one needs a signed-in session.
**Your password is never needed and should never be given to a script** — Yahoo
2FA-blocks scripted logins, so it would fail anyway. Two working routes:

### Reuse your Firefox login (no password)

Firefox stores cookies unencrypted, so an existing Yahoo login can be reused directly.
Chrome encrypts its cookie store with a per-user key, so this is Firefox-only.

```sh
./value_screener.py --yahoo-status     # is a reusable login present?
./value_screener.py --from-yahoo       # screen the signed-in portfolio
```

If it reports no session, log in at finance.yahoo.com in Firefox and re-run. The
signed-out page returns HTTP 200 rather than a 401, so the session is checked *before*
parsing — otherwise "signed out" and "empty portfolio" would look identical.

### Or export the CSV (no account access at all)

Yahoo Finance → your portfolio → ⋮ → **Export**. This route is better in one respect:
the export carries **quantity and cost basis**, which a watchlist does not.

```sh
./value_screener.py --portfolio ~/Downloads/quotes.csv
```

Columns are matched by name, not position, because Yahoo reorders them between the
portfolio and watchlist views.

## Holdings-aware output

When the source carries quantity, a second table answers the holder's question, which
is not the buyer's question — a stock can be simultaneously "don't buy more" and "worth
keeping":

```
HOLDINGS        VALUE  WEIGHT      P/L  ACTION WHY
GILD           22,517  20.2%     121%  EXIT   unprofitable (negative trailing EPS)
KO             26,475  23.8%      96%  TRIM   72% above estimated fair value
MO             27,808  25.0%      55%  HOLD   fair
T              12,700  11.4%      40%  ADD    still 36% below fair value
TOTAL         111,402
```

| Action | Trigger |
|---|---|
| `EXIT` | Failed a quality gate — regardless of gain |
| `TRIM` | >25% above estimated fair value |
| `HOLD` | Neither |
| `ADD` | Still screens `BUY` |

`GILD` above is the case a buy-only screener never shows you: **up 121% and flagged
EXIT**. The gain is exactly what makes that hard to see.

The sell threshold (25%) is deliberately wider than the buy threshold, because
declining to buy is free while selling costs spread and tax. **Tax is not modelled** —
a `TRIM` on a long-held position may well be wrong after capital gains.

## Tuning

Defaults are conservative on purpose. The discount rate is the input a DCF is most
sensitive to, so it is a flag rather than a buried constant.

```sh
--buy-margin 0.4          # demand a 40% discount
--risk-free 0.045         # risk-free rate (default 4%)
--equity-premium 0.06     # equity risk premium (default 5%)
--terminal-growth 0.02    # perpetual growth after year 10 (default 2.5%)
--max-debt-to-equity 150  # leverage gate, percent (default 200 = 2.0x)
--no-cache                # refetch everything
```

## Known limits

- **Trailing EPS is TTM**, so a one-off writedown flips a healthy company to `AVOID`.
  Gilead screens as unprofitable on a single impairment charge. Read the `--explain`
  output before believing a verdict.
- **Banks and REITs** get no DCF — free cash flow is not meaningful when lending is an
  investing outflow or the business is capital expenditure. The cash-flow and leverage
  gates are suspended for those two sectors; the profitability gate still applies.
- **Cyclicals** look cheapest at the top of their cycle, when trailing earnings peak.
  Falling estimates now veto a `BUY`, which catches the case where analysts have already
  noticed. It will not catch a cycle turning that nobody has forecast yet.
- **Analyst estimates are optimistic**, most of all for companies in trouble. They join
  the growth `min()` rather than replacing history precisely so they can only ever lower
  a valuation, never inflate one.
- **Yahoo data is free and imperfect.** `info["freeCashflow"]` disagreed with Microsoft's
  own filed cash flow statement by 4x, which is why the annual statement is preferred and
  the summary field is only a fallback.
- Everything assumes the listing currency; no FX normalisation across a mixed watchlist.


## Tracking changes over time

A single screen says "is this cheap now". Most of what matters is the derivative: what
crossed into `BUY` this week, and whether a discount widened because the price fell or
because the estimate rose. Those call for opposite reactions and look identical in the
margin alone.

```sh
./value_screener.py --record      # store this run
./value_screener.py --changes     # what moved since each ticker's last recorded run
./value_screener.py --trend EQT   # one ticker's timeline
```

```
SINCE LAST RUN
  EQT        FAIR -> WATCH   wider on a -11% price fall
  DIS        BUY -> WATCH    narrower on a 47% price rise
```

Verdict crossings are always listed; margin drift only above 5%, so a quote wobble is
not reported as news. Comparison is per ticker against *its own* last reading, not
against one global previous run — watchlists get edited, and a symbol screened a month
ago should still be comparable rather than silently dropped.

Stored in `history.db` (SQLite, gitignored — it records what you screen).

**Why SQLite and not the vector store?** Every question here is exact: one ticker on
one date, margins above a threshold, the row before this one. Those are key lookups,
range scans and ordering. Approximate nearest-neighbour search answers none of them,
and embedding a row of floats to retrieve it by similarity would be slower, lossier and
dependent on a model. Vectors answer a different question — see below.

## Semantic index (optional)

The question a vector store *does* answer is "what else is like this", so what gets
embedded is the company's **business description**, never its numbers. The last verdict
rides along as metadata, which is the useful combination: find businesses similar to
one that screens well, and see immediately whether they screen well too.

```sh
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-vector.txt

./value_screener.py --portfolio list.csv --index   # screen and index
./value_screener.py --similar NVDA                 # businesses like this one
./value_screener.py --find "power semiconductors"  # plain-language search
./value_screener.py --indexed                      # everything stored
```

```
Businesses most like NVDA:
  TICKER       SIM       PRICE  MARGIN  VERDICT    NAME
  INTC        0.79      108.60      --  AVOID      Intel Corporation
  AMD         0.75      559.82   -529%  EXPENSIVE  Advanced Micro Devices, Inc.
  OTEX        0.67       22.58     34%  WATCH      Open Text Corporation
```

Searching *"semiconductor manufacturing equipment"* returns AMAT and ASML because of
what they do, not because a sector string matched. This is the direct answer to a screen
that finds nothing: widen the universe rather than lower the threshold.

Companies with no business description are skipped rather than embedded from their
ticker symbol, and funds are skipped too — an ETF's blurb describes a strategy, so
indexing it would return a tracker whenever you searched for the industry it tracks.

Stored in `screen_index.db` (Milvus Lite, embedded, no server — gitignored). Model,
dimension and metric match `congress-trades/search_common.py` deliberately, so two
indexes on one machine share an embedding space. The dependencies are optional and
imported lazily; the valuation models never touch them.

## Daily Discord post at the opening bell

`market_open_monitor.py` screens `watchlist.json` and posts to Discord when the New
York market opens — only on days it actually opens.

```sh
./market_open_monitor.py --dry-run                        # print, don't post
./market_open_monitor.py --force                          # post now, ignoring guards
./market_open_monitor.py --now 2026-09-21T13:30:00+00:00 --dry-run   # simulate a bell
```

Crontab (installed):

```
*/15 13-15 * * 1-5 /usr/bin/python3 .../market_open_monitor.py >> .../market_open.log 2>&1
```

**Cron polls; the script decides.** A fixed local hour would be wrong for about four
weeks a year: this machine runs Europe/Dublin, the exchange runs America/New_York, and
they switch daylight saving on different dates. The bell is 14:30 Dublin most of the
year but 13:30 between the US and EU switchover dates, so cron covers both hours and
`market_clock` resolves the real one through `zoneinfo`.

Three guards stop a polling schedule becoming a spamming one:

- **Trading day.** Weekends and the full NYSE holiday calendar, computed rather than
  fetched — including Good Friday (Easter-derived) and weekend observance, so 4 July
  2026 closes on Friday the 3rd. Posting "market open" on Thanksgiving teaches you to
  ignore the notification.
- **Opening window.** Within 45 minutes *after* the bell, never before — firing early
  would report yesterday's close as today's price.
- **Once per session.** A state file records the exchange date posted, so the other
  cron ticks are no-ops. A `flock` makes an overrunning run skip rather than double-post.

The message leads with **what changed** — verdict crossings since the last recorded run
— then what is currently below fair value. A list of verdicts is the same most mornings
and gets skimmed into invisibility; a crossing is the reason to open the notification.

```
**Market open** - 2026-09-21 - 33 screened

__Changed since last run__
- **EQT** FAIR -> WATCH (wider on a -11% price fall)

__Below estimated fair value__
- **NOVO-B.CO** 281.55 vs 489.06 est. (42%) WATCH - analysts span 90% of price
- **OTEX** 22.58 vs 34.45 est. (34%) WATCH - estimates falling (3 down vs 0 up in 30d)

_6 failing quality gates: CL, IEP, INTC, IONQ, PEP, QUBT_
```

Webhook is read from `~/.config/fare-monitor/config.json`, shared with the other
monitors on this machine. `--now` exists because a time-gated job that can only be
exercised at 14:30 on a weekday is a job whose scheduling fails silently.

Note the calendar is the **NYSE** one. A non-US holding like `NOVO-B.CO` is screened on
New York's schedule, not Copenhagen's.

## Caching

Three tiers, matching how fast each kind of data actually moves: quotes 1h, analyst
estimates 24h, annual statements 7 days, all in `cache/`. One TTL for all three would
either serve stale prices or refetch five years of financials to learn a stock moved
twelve cents. `--no-cache` bypasses all three.

## Tests

```sh
python3 -m unittest discover -s tests
```

No network. They pin the ways a screener produces a confident wrong number: growth
invented from a negative base, Gordon growth dividing by a negative denominator, NaN
sliding through the thresholds, and a large discount derived from contradictory anchors.

## Your data stays out of this repo

This repo is public; your positions are not. Gitignored: `watchlist.json`, any `*.csv`
(portfolio exports carry cost basis), `portfolio*.json`, and `cache/` (fetched
fundamentals for whatever you screened). Only `watchlist.example.json` is tracked.

No credentials are ever read or stored. The optional Yahoo path reuses an existing
Firefox session cookie; the CSV path needs no account access at all.

## Not investment advice

Mechanical output from free data and four openly-wrong models. Every number is an
estimate with an error bar the screener cannot compute.
