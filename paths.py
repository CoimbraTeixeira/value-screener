"""Where the screener keeps the files it writes, resolved once for every module.

By default that is the repo itself -- `cache/`, `state/`, `history.db`, `watchlist.json`
and `screen_index.db` sit next to the code, gitignored, exactly as cron has always used
them. `VALUE_SCREENER_DATA_DIR` moves all of them somewhere else in one go, so a test
harness can run the real scripts against a throwaway directory without reading or
clobbering the live cache, state and history that the hourly jobs depend on.

One module rather than a constant per file: five modules each deriving their own base
from `__file__` is how one of them ends up writing into the live repo while the others
write into the sandbox.
"""

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent


def resolve_data_dir(environ=os.environ) -> Path:
    """The base for every generated file: the override if set, else the repo.

    An empty value counts as unset, so `VALUE_SCREENER_DATA_DIR= ./script` cannot
    silently mean "the current working directory".
    """
    override = environ.get("VALUE_SCREENER_DATA_DIR")
    return Path(override).expanduser() if override else REPO_DIR


DATA_DIR = resolve_data_dir()
