"""Persistent record of what each run concluded, so verdicts can be compared over time.

A single screen is a snapshot and answers "is this cheap now". Most of what is worth
knowing is a derivative: which stock crossed into BUY this week, whether a discount is
widening because the estimate held and the price fell, or because the estimate is being
cut out from under it. Those need yesterday's numbers, which the screener previously
threw away.

SQLite rather than the vector store next door. Every question here is exact -- one
ticker on one date, margins above a threshold, the row before this one -- and those are
key lookups, range scans and ordering. Approximate nearest-neighbour search answers none
of them, and embedding a row of floats to retrieve it by similarity would be slower,
lossier and dependent on a model. Vectors earn their place when the query is "what else
is like this", which is a question about business descriptions, not about prices.
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
HISTORY_PATH = REPO_DIR / "history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    ticker     TEXT NOT NULL,
    run_at     TEXT NOT NULL,
    price      REAL NOT NULL,
    currency   TEXT,
    fair_value REAL,
    margin     REAL,
    verdict    TEXT NOT NULL,
    anchors    TEXT,
    flags      TEXT,
    gates      TEXT,
    PRIMARY KEY (ticker, run_at)
);
CREATE INDEX IF NOT EXISTS snapshots_by_ticker ON snapshots (ticker, run_at DESC);
CREATE INDEX IF NOT EXISTS snapshots_by_run ON snapshots (run_at DESC);
"""


def connect(path: Path = HISTORY_PATH) -> sqlite3.Connection:
    """Open the history store, creating the schema if this is the first run."""
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def record(connection: sqlite3.Connection, results, run_at: str | None = None) -> str:
    """Persist one run. Returns the run timestamp, which identifies the batch.

    A single timestamp for the whole batch rather than per-row: the rows are one
    observation of a watchlist at one moment, and per-row times would make "the previous
    run" ambiguous when a fetch takes a few seconds.
    """
    stamp = run_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    connection.executemany(
        "INSERT OR REPLACE INTO snapshots "
        "(ticker, run_at, price, currency, fair_value, margin, verdict, anchors, flags, gates) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(r.ticker, stamp, r.price, r.currency, r.fair_value, r.margin_of_safety,
          r.verdict,
          json.dumps({a.name: a.value for a in r.anchors}),
          json.dumps(r.flags), json.dumps(r.gate_failures))
         for r in results])
    connection.commit()
    return stamp


@dataclass
class Change:
    """One ticker's movement between two runs."""

    ticker: str
    previous_at: str
    verdict_from: str
    verdict_to: str
    price_from: float
    price_to: float
    margin_from: float | None
    margin_to: float | None

    @property
    def verdict_changed(self) -> bool:
        return self.verdict_from != self.verdict_to

    @property
    def price_move(self) -> float | None:
        if self.price_from:
            return self.price_to / self.price_from - 1.0
        return None

    @property
    def margin_move(self) -> float | None:
        if self.margin_from is None or self.margin_to is None:
            return None
        return self.margin_to - self.margin_from

    def describe(self) -> str:
        """Why the margin moved, which is the part a bare number hides.

        A discount can widen because the price fell (the market changed its mind) or
        because the fair value rose (the business got better). Those call for opposite
        reactions, and the two are indistinguishable from the margin alone.
        """
        move, price = self.margin_move, self.price_move
        if move is None or price is None:
            return ""
        if abs(move) < 0.01:
            return "unchanged"
        direction = "wider" if move > 0 else "narrower"
        if move > 0 and price < -0.01:
            return f"{direction} on a {price:.0%} price fall"
        if move > 0 and price >= -0.01:
            return f"{direction} on a higher estimate"
        if move < 0 and price > 0.01:
            return f"{direction} on a {price:.0%} price rise"
        return f"{direction} on a lower estimate"


def previous_run(connection: sqlite3.Connection, ticker: str,
                 before: str) -> sqlite3.Row | None:
    """The most recent stored snapshot for a ticker strictly before a timestamp."""
    return connection.execute(
        "SELECT * FROM snapshots WHERE ticker = ? AND run_at < ? "
        "ORDER BY run_at DESC LIMIT 1", (ticker, before)).fetchone()


def changes(connection: sqlite3.Connection, results, before: str) -> list[Change]:
    """What moved since each ticker's last recorded run.

    Compared per ticker rather than against one global previous run, because a watchlist
    is edited: a symbol added today has no prior row, and a symbol screened last month
    should still be comparable to that month-old reading rather than silently dropped.
    """
    moved = []
    for result in results:
        prior = previous_run(connection, result.ticker, before)
        if prior is None:
            continue
        moved.append(Change(
            ticker=result.ticker,
            previous_at=prior["run_at"],
            verdict_from=prior["verdict"],
            verdict_to=result.verdict,
            price_from=prior["price"],
            price_to=result.price,
            margin_from=prior["margin"],
            margin_to=result.margin_of_safety,
        ))
    return moved


def trend(connection: sqlite3.Connection, ticker: str,
          limit: int = 20) -> list[sqlite3.Row]:
    """A ticker's recorded history, oldest first so it reads as a timeline."""
    rows = connection.execute(
        "SELECT * FROM snapshots WHERE ticker = ? ORDER BY run_at DESC LIMIT ?",
        (ticker.upper(), limit)).fetchall()
    return list(reversed(rows))


def run_count(connection: sqlite3.Connection) -> int:
    return connection.execute(
        "SELECT COUNT(DISTINCT run_at) FROM snapshots").fetchone()[0]
