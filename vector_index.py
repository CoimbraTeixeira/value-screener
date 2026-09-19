"""Semantic index of screened companies: what each business does, plus its last verdict.

This is the half of persistence that genuinely wants vectors. Exact history -- one
ticker on one date, margins over a threshold -- lives in history.py on SQLite, because
those are key lookups and range scans. What a vector store answers instead is "what else
is like this", and that question only has an answer if the thing embedded is prose.

So the vector is the company's *business description*, never its numbers. Embedding a
row of floats would be a slower, lossier way to do a lookup SQLite already does exactly.
Embedding "designs and manufactures analogue power semiconductors" lets a search for
power management retrieve MPWR because of what it does, not because a sector string
happened to contain a matching word.

The last verdict rides along as scalar metadata. That combination is the useful one:
find businesses similar to a company that screens well, and see immediately whether they
screen well too.

Model, dimension and metric match congress-trades/search_common.py deliberately. Two
indexes on one machine using different embedding spaces for the same tickers would be a
trap for anyone who later tried to compare them.

Heavy imports (sentence-transformers/torch, pymilvus) are deferred to call time so the
core screener stays importable and fast without them installed.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
INDEX_PATH = REPO_DIR / "screen_index.db"      # Milvus Lite store (a directory)
COLLECTION = "screened"

# Same embedding space as the congress-trades index, so the two are comparable.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
METRIC = "COSINE"
# bge wants an instruction prefix on the query side only; documents are embedded bare.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Business summaries run to several hundred words and the model truncates anyway; this
# keeps the stored text bounded without cutting into the part that describes the business.
MAX_SUMMARY_CHARS = 2000

_model = None


class VectorIndexUnavailable(RuntimeError):
    """Raised when the optional vector dependencies are not installed.

    A distinct type so the CLI can print an install hint rather than a stack trace: the
    vector half is optional by design and its absence is a configuration state, not a
    bug.
    """


def _load_model():
    """The embedding model, loaded once per process.

    Loading costs a few seconds and a few hundred megabytes, which is why nothing here
    imports it at module scope and why lookups that only read stored vectors never call
    it at all.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise VectorIndexUnavailable(
                "sentence-transformers is not installed. "
                "pip install -r requirements-vector.txt") from exc
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _client():
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise VectorIndexUnavailable(
            "pymilvus is not installed. pip install -r requirements-vector.txt") from exc
    return MilvusClient(uri=str(INDEX_PATH))


def _ensure_collection(client) -> None:
    """Create the collection on first use, and load it into memory for querying.

    The schema is explicit rather than dynamic so a verdict written as a float, or a
    ticker written as an int, fails at insert instead of silently poisoning a later
    filter expression.

    The load matters on every call, not just creation: Milvus leaves a collection
    'released' when reopened in a new process, and search against a released collection
    raises rather than returning empty -- so a second run of the CLI would fail where
    the first succeeded.
    """
    from pymilvus import DataType

    if client.has_collection(COLLECTION):
        client.load_collection(COLLECTION)
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("ticker", DataType.VARCHAR, is_primary=True, max_length=32)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)
    schema.add_field("name", DataType.VARCHAR, max_length=128)
    schema.add_field("sector", DataType.VARCHAR, max_length=64)
    schema.add_field("verdict", DataType.VARCHAR, max_length=16)
    schema.add_field("summary", DataType.VARCHAR, max_length=MAX_SUMMARY_CHARS)
    schema.add_field("currency", DataType.VARCHAR, max_length=8)
    schema.add_field("updated_at", DataType.VARCHAR, max_length=32)
    schema.add_field("price", DataType.DOUBLE)
    schema.add_field("fair_value", DataType.DOUBLE)
    schema.add_field("margin", DataType.DOUBLE)

    index = client.prepare_index_params()
    index.add_index(field_name="vector", index_type="FLAT", metric_type=METRIC)
    client.create_collection(COLLECTION, schema=schema, index_params=index)


def embed(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Normalised embeddings. Queries get bge's instruction prefix, documents do not."""
    model = _load_model()
    prepared = [QUERY_PREFIX + t for t in texts] if is_query else texts
    return model.encode(prepared, normalize_embeddings=True).tolist()


@dataclass
class Neighbour:
    ticker: str
    name: str
    sector: str
    verdict: str
    price: float
    fair_value: float
    margin: float
    similarity: float
    currency: str = "USD"


def upsert(results, summaries: dict[str, str]) -> int:
    """Store or refresh each screened company's description and latest verdict.

    Companies without a business description are skipped rather than embedded from their
    ticker symbol: a vector built from the string "OTEX" encodes nothing about the
    business and would pollute every neighbour search that touched it. Funds are skipped
    for the same reason -- an ETF's description is about a strategy, not a company.
    """
    usable = [r for r in results if summaries.get(r.ticker)]
    if not usable:
        return 0

    client = _client()
    _ensure_collection(client)
    vectors = embed([summaries[r.ticker][:MAX_SUMMARY_CHARS] for r in usable])
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    client.upsert(COLLECTION, [{
        "ticker": r.ticker,
        "vector": vector,
        "name": (r.name or r.ticker)[:128],
        "sector": (getattr(r, "sector", "") or "")[:64],
        "verdict": r.verdict[:16],
        "summary": summaries[r.ticker][:MAX_SUMMARY_CHARS],
        "currency": (r.currency or "USD")[:8],
        "updated_at": stamp,
        "price": float(r.price),
        # Milvus has no null for scalars, so absent values are stored as a sentinel the
        # formatter recognises rather than as 0.0, which would read as a real estimate.
        "fair_value": float(r.fair_value) if r.fair_value else -1.0,
        "margin": float(r.margin_of_safety) if r.margin_of_safety is not None else -99.0,
    } for r, vector in zip(usable, vectors)])
    return len(usable)


def _to_neighbours(rows, drop: str | None = None) -> list[Neighbour]:
    found = []
    for row in rows:
        entity = row.get("entity", row)
        ticker = entity.get("ticker", "")
        if drop and ticker == drop:
            continue
        found.append(Neighbour(
            ticker=ticker,
            name=entity.get("name", ""),
            sector=entity.get("sector", ""),
            verdict=entity.get("verdict", ""),
            price=entity.get("price", 0.0),
            fair_value=entity.get("fair_value", -1.0),
            margin=entity.get("margin", -99.0),
            currency=entity.get("currency", "USD"),
            # Milvus returns cosine distance as a similarity for COSINE metric.
            similarity=row.get("distance", 0.0),
        ))
    return found


OUTPUT_FIELDS = ["ticker", "name", "sector", "verdict", "price", "fair_value",
                 "margin", "currency"]


def similar(ticker: str, limit: int = 8) -> list[Neighbour]:
    """Indexed companies whose business most resembles this one.

    Reuses the stored vector rather than re-embedding, so this costs no model load: the
    description was already embedded when the ticker was indexed.
    """
    client = _client()
    _ensure_collection(client)
    symbol = ticker.upper().strip()
    rows = client.query(COLLECTION, filter=f'ticker == "{symbol}"',
                        output_fields=["vector"], limit=1)
    if not rows:
        raise LookupError(f"{symbol} is not indexed. Screen it with --index first.")

    hits = client.search(COLLECTION, data=[rows[0]["vector"]], limit=limit + 1,
                         output_fields=OUTPUT_FIELDS)
    return _to_neighbours(hits[0] if hits else [], drop=symbol)


def search(text: str, limit: int = 8) -> list[Neighbour]:
    """Indexed companies matching a plain-language description of a business.

    This is the query that justifies the whole index: "power semiconductors" should
    retrieve the companies that make them, whatever their sector string says.
    """
    client = _client()
    _ensure_collection(client)
    vector = embed([text], is_query=True)[0]
    hits = client.search(COLLECTION, data=[vector], limit=limit,
                         output_fields=OUTPUT_FIELDS)
    return _to_neighbours(hits[0] if hits else [])


def stored() -> list[Neighbour]:
    """Everything currently indexed, for `--indexed`."""
    client = _client()
    _ensure_collection(client)
    rows = client.query(COLLECTION, filter='ticker != ""',
                        output_fields=OUTPUT_FIELDS, limit=10_000)
    return sorted(_to_neighbours(rows), key=lambda n: n.ticker)
