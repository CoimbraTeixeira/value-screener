"""Tests for the environment seams that let the real scripts run against fakes.

Run with: python3 -m unittest discover -s tests

Three overrides exist so a harness can run the screener end to end with no real side
effects: VALUE_SCREENER_DATA_DIR moves every generated file, MONITOR_CONFIG points the
Discord post at another webhook, and VALUE_SCREENER_OFFLINE=1 replays recorded cache
files instead of calling Yahoo. These pin both halves of each: unset means exactly the
paths and behaviour cron has always had, and set means nothing leaks back to the live
cache, state, history or webhook. Offline mode in particular is proven by making any
yfinance call fail the test, not by trusting that none happened.

No network: the webhook is a stdlib server on 127.0.0.1, and the end-to-end runs poison
the proxy settings so a stray Yahoo request would fail fast rather than succeed.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import market_data  # noqa: E402
import paths  # noqa: E402
import value_screener  # noqa: E402

SEAM_VARS = ("VALUE_SCREENER_DATA_DIR", "VALUE_SCREENER_OFFLINE", "MONITOR_CONFIG")

QUOTE = {"currentPrice": 87.6, "regularMarketPrice": 87.55, "previousClose": 87.77,
         "shortName": "Coca-Cola Company (The)", "currency": "USD", "quoteType": "EQUITY",
         "sector": "Consumer Defensive", "longBusinessSummary": "Makes soft drinks.",
         "trailingEps": 3.04, "bookValue": 7.1, "sharesOutstanding": 4.3e9,
         "dividendRate": 2.04, "payoutRatio": 0.67, "returnOnEquity": 0.4,
         "debtToEquity": 150.0, "fetched_at": "2026-01-02T15:00:00Z"}
STATEMENTS = {"fcf_history": [9.5e9, 9.7e9, 4.7e9, 5.3e9], "eps_history": [2.19, 2.47, 2.46, 3.04],
              "historical_pe": [22.6, 24.1, 22.4, 26.0]}
FORWARD = {"eps_next_year": 3.52, "analyst_count": 24, "revisions_up": 20, "revisions_down": 1,
           "share_growth": 0.0006}

# Far older than every TTL, so a cache read in normal mode would be a miss.
STALE = time.time() - 365 * 86400


def clean_env(**overrides) -> dict:
    """The current environment minus the seams (so "unset" really is unset even inside
    a harness that set them), minus proxies, plus the given overrides."""
    env = {k: v for k, v in os.environ.items()
           if k not in SEAM_VARS and "proxy" not in k.lower()}
    env.update(overrides)
    return env


def record(directory: Path, ticker: str = "KO", kinds=("quote", "statements", "forward"),
           mtime: float = STALE) -> None:
    """Write recorded cache files in the format market_data itself writes."""
    payloads = {"quote": QUOTE, "statements": STATEMENTS, "forward": FORWARD}
    directory.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        path = directory / f"{ticker}.{kind}.json"
        path.write_text(json.dumps(payloads[kind]))
        os.utime(path, (mtime, mtime))


def resolved_paths(env: dict) -> dict:
    """Every generated-file location, as a fresh interpreter resolves it under `env`.

    A subprocess rather than importlib.reload: the constants are fixed at import, and
    that import-time resolution is exactly what cron exercises.
    """
    probe = (
        "import json, history, market_data, market_open_monitor, value_screener, vector_index\n"
        "print(json.dumps({\n"
        "  'cache': str(market_data.CACHE_DIR),\n"
        "  'history': str(history.HISTORY_PATH),\n"
        "  'state': str(market_open_monitor.STATE_DIR),\n"
        "  'lock': str(market_open_monitor.LOCK_PATH),\n"
        "  'watchlist': str(value_screener.WATCHLIST_PATH),\n"
        "  'example': str(value_screener.EXAMPLE_WATCHLIST_PATH),\n"
        "  'index': str(vector_index.INDEX_PATH),\n"
        "  'config': str(value_screener.CONFIG_PATH),\n"
        "}))\n")
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


class FakeWebhook:
    """A loopback Discord stand-in: records each JSON body, answers 204."""

    def __init__(self):
        posts = self.posts = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                posts.append((self.path, json.loads(self.rfile.read(length))))
                self.send_response(204)
                self.end_headers()

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/webhooks/1/fake"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class DataDir(unittest.TestCase):
    def test_unset_means_the_repo(self):
        self.assertEqual(paths.resolve_data_dir({}), REPO)

    def test_empty_means_the_repo_not_the_working_directory(self):
        self.assertEqual(paths.resolve_data_dir({"VALUE_SCREENER_DATA_DIR": ""}), REPO)

    def test_override_is_honoured(self):
        self.assertEqual(paths.resolve_data_dir({"VALUE_SCREENER_DATA_DIR": "/tmp/sandbox"}),
                         Path("/tmp/sandbox"))

    def test_unset_leaves_every_file_where_cron_expects_it(self):
        found = resolved_paths(clean_env())
        self.assertEqual(found, {
            "cache": str(REPO / "cache"),
            "history": str(REPO / "history.db"),
            "state": str(REPO / "state"),
            "lock": str(REPO / "state" / "market_open.lock"),
            "watchlist": str(REPO / "watchlist.json"),
            "example": str(REPO / "watchlist.example.json"),
            "index": str(REPO / "screen_index.db"),
            "config": str(Path.home() / ".config" / "fare-monitor" / "config.json"),
        })

    def test_override_moves_every_generated_file_and_only_those(self):
        sandbox = Path(tempfile.mkdtemp())
        found = resolved_paths(clean_env(VALUE_SCREENER_DATA_DIR=str(sandbox)))
        example = found.pop("example")
        config = found.pop("config")
        for name, location in found.items():
            with self.subTest(name=name):
                self.assertTrue(Path(location).is_relative_to(sandbox), location)
        # The template is code and the webhook config has its own seam; neither moves.
        self.assertEqual(example, str(REPO / "watchlist.example.json"))
        self.assertEqual(config, str(Path.home() / ".config" / "fare-monitor" / "config.json"))


class MonitorConfig(unittest.TestCase):
    def test_unset_means_the_shared_config(self):
        self.assertEqual(value_screener.resolve_config_path({}),
                         Path.home() / ".config" / "fare-monitor" / "config.json")

    def test_empty_means_the_shared_config(self):
        self.assertEqual(value_screener.resolve_config_path({"MONITOR_CONFIG": ""}),
                         value_screener.DEFAULT_CONFIG_PATH)

    def test_override_is_honoured(self):
        self.assertEqual(value_screener.resolve_config_path({"MONITOR_CONFIG": "/tmp/c.json"}),
                         Path("/tmp/c.json"))

    def test_market_open_delivery_posts_to_the_overridden_webhook(self):
        """The morning post goes through value_screener.notify; this proves that path
        reads MONITOR_CONFIG in a real process and reaches the configured URL."""
        config = Path(tempfile.mkdtemp()) / "config.json"
        with FakeWebhook() as hook:
            config.write_text(json.dumps({"webhook_url": hook.url}))
            subprocess.run([sys.executable, "-c",
                            "from market_open_monitor import notify; notify('hello')"],
                           cwd=REPO, env=clean_env(MONITOR_CONFIG=str(config)),
                           capture_output=True, text=True, check=True)
        self.assertEqual(hook.posts, [("/api/webhooks/1/fake", {"content": "hello"})])

    def test_a_missing_override_fails_instead_of_falling_back_to_the_real_webhook(self):
        missing = Path(tempfile.mkdtemp()) / "absent.json"
        run = subprocess.run([sys.executable, "-c",
                              "from value_screener import notify; notify('hello')"],
                             cwd=REPO, env=clean_env(MONITOR_CONFIG=str(missing)),
                             capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn(str(missing), run.stderr)


class Offline(unittest.TestCase):
    def setUp(self):
        self.cache = Path(tempfile.mkdtemp()) / "cache"
        patcher = mock.patch.object(market_data, "CACHE_DIR", self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        # Any construction of a yfinance handle fails the test outright.
        ticker = mock.patch.object(market_data.yf, "Ticker",
                                   side_effect=AssertionError("yf.Ticker called offline"))
        self.ticker = ticker.start()
        self.addCleanup(ticker.stop)

    def offline(self, value="1"):
        return mock.patch.dict(os.environ, {"VALUE_SCREENER_OFFLINE": value})

    def test_stale_recordings_are_replayed_without_yahoo(self):
        record(self.cache)
        before = {p.name: p.stat().st_mtime for p in self.cache.iterdir()}
        with self.offline():
            fundamentals = market_data.fetch("ko")
        self.ticker.assert_not_called()
        self.assertEqual(fundamentals.ticker, "KO")
        self.assertEqual(fundamentals.price, 87.6)
        self.assertEqual(fundamentals.fcf_history, STATEMENTS["fcf_history"])
        self.assertEqual(fundamentals.forward.analyst_count, 24)
        # Replaying must not rewrite the recording.
        self.assertEqual({p.name: p.stat().st_mtime for p in self.cache.iterdir()}, before)

    def test_a_missing_statement_recording_is_an_error_not_a_fetch(self):
        record(self.cache, kinds=("quote", "forward"))
        with self.offline(), self.assertRaisesRegex(ValueError, "no recorded data"):
            market_data.fetch("KO")
        self.ticker.assert_not_called()

    def test_a_missing_quote_recording_is_an_error_not_a_fetch(self):
        with self.offline(), self.assertRaisesRegex(ValueError, "no recorded data"):
            market_data.fetch("NOPE")
        self.ticker.assert_not_called()

    def test_forward_is_required_only_when_asked_for(self):
        record(self.cache, kinds=("quote", "statements"))
        with self.offline():
            self.assertIsNone(market_data.fetch("KO", with_forward=False).forward)
            with self.assertRaisesRegex(ValueError, "no recorded data"):
                market_data.fetch("KO")
        self.ticker.assert_not_called()

    def test_unset_still_treats_stale_cache_as_a_miss(self):
        """Default behaviour is unchanged: a year-old quote goes back to Yahoo."""
        record(self.cache)
        env = clean_env()
        with mock.patch.dict(os.environ, env, clear=True), \
                self.assertRaisesRegex(AssertionError, "yf.Ticker called"):
            market_data.fetch("KO")
        self.ticker.assert_called_once_with("KO")

    def test_only_the_value_1_switches_it_on(self):
        record(self.cache)
        with self.offline("0"), self.assertRaisesRegex(AssertionError, "yf.Ticker called"):
            market_data.fetch("KO")

    def test_unset_serves_fresh_cache_without_yahoo_as_before(self):
        record(self.cache, mtime=time.time())
        with mock.patch.dict(os.environ, clean_env(), clear=True):
            self.assertEqual(market_data.fetch("KO").price, 87.6)
        self.ticker.assert_not_called()


class EndToEnd(unittest.TestCase):
    def test_market_open_monitor_runs_entirely_inside_the_sandbox(self):
        """The real script, every seam set: recorded data in, one post to the fake out,
        state and history written to the sandbox and nowhere else."""
        sandbox = Path(tempfile.mkdtemp())
        record(sandbox / "cache")
        (sandbox / "watchlist.json").write_text(json.dumps(["KO"]))
        config = sandbox / "config.json"
        with FakeWebhook() as hook:
            config.write_text(json.dumps({"webhook_url": hook.url}))
            # A dead proxy for everything but loopback: should offline mode ever leak a
            # Yahoo request, it fails here instead of quietly reaching the internet.
            env = clean_env(VALUE_SCREENER_DATA_DIR=str(sandbox), VALUE_SCREENER_OFFLINE="1",
                            MONITOR_CONFIG=str(config),
                            HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                            NO_PROXY="127.0.0.1,localhost")
            run = subprocess.run([sys.executable, "market_open_monitor.py", "--force",
                                  "--now", "2026-09-21T13:35:00+00:00"],
                                 cwd=REPO, env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(len(hook.posts), 1)
        content = hook.posts[0][1]["content"]
        self.assertIn("Market open", content)
        self.assertIn("KO", content)
        self.assertNotIn("could not be fetched", content)
        state = json.loads((sandbox / "state" / "market_open_posted.json").read_text())
        self.assertEqual(state["last_session"], "2026-09-21")
        self.assertTrue((sandbox / "history.db").exists())


if __name__ == "__main__":
    unittest.main()
