"""all-MiniLM-L6-v2 loader as a module-level singleton, CPU only.

Never instantiate per request — an 8 GB RAM budget cannot absorb it
(PROJECT_PLAN.md §1.1 rule 4, §2.4).

This module is the ONLY place the embedding model is constructed. db/seed.py imports
embed_texts() from here rather than loading its own copy; Phase 2 appends the pgvector
cosine search (top-k = 3) to this same module so the write path and the read path can
never end up on different models or different normalization.
"""

from functools import lru_cache
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from app.config import settings

# Cap intra-op threads before any model work. PostgreSQL runs on the same 8 GB machine
# and torch otherwise spawns one thread per core, which thrashes against it
# (§1.1 rule 6).
torch.set_num_threads(4)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Return the process-wide MiniLM instance, loading it on first call.

    lru_cache is the singleton: the second and every later call returns the same object.
    Loading is lazy rather than at import so that importing this module (for type hints,
    or from a test that never embeds anything) does not pay ~90 MB of RAM.

    device="cpu" is not a default worth relying on — it is a hard constraint. The
    4 GB GTX 1650 is reserved entirely for Ollama, and every non-LLM model in this
    project runs on CPU (§1.1 rule 5). Putting MiniLM on the GPU would evict LLM layers
    into system RAM and drop generation throughput 3-5x.
    """
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    model.eval()  # inference only; no dropout, no autograd bookkeeping
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings into 384-dim vectors ready for pgvector insertion.

    Batch in one call wherever possible — the per-call overhead is model dispatch, not
    per-string work, and a loop of single-string calls is measurably slower for no
    memory saving.

    normalize_embeddings=True matters beyond tidiness: policy_chunks is indexed with
    ivfflat vector_cosine_ops, so stored vectors and query vectors must be normalized
    the same way. Normalizing here, at the single write path, is what guarantees that.
    """
    if not texts:
        return []

    with torch.no_grad():
        vectors = get_embedder().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    # psycopg/pgvector wants plain Python sequences, not numpy rows.
    return [vector.tolist() for vector in vectors]


# Top-k is 3, fixed by §2.4. It is a context-window decision as much as a relevance one:
# the 4096-token budget (§1.1 rule 2) has to hold the system prompt, the SQL tool's rows,
# and the answer as well as these chunks.
DEFAULT_TOP_K = 3


def search_policy_chunks(query: str, k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Cosine-similarity search over `policy_chunks`, nearest first (§2.4).

    Returns dicts of `{chunk_id, doc_name, chunk_text, distance}`.

    `<=>` is pgvector's cosine *distance* operator: 0.0 is identical, 2.0 is opposite.
    Distance rather than similarity is returned because that is what the operator gives
    and what the index is built on (`vector_cosine_ops` in db/schema.sql) — converting to
    a similarity score here would invite a caller to compare it against a threshold that
    was never calibrated.

    The query vector is embedded through `embed_texts` above, so it is normalized exactly
    the way the stored vectors were at seed time. That shared path is the point: cosine
    distance between differently-normalized vectors is quietly wrong rather than loudly
    broken, and it would surface as mediocre retrieval that nobody traces back to here.

    No `rm_id`, and no entitlement scoping. `policy_chunks` holds the bank's own six
    policy documents — no customer_id, no join path to `customers`, identical content for
    every RM. That exemption is documented in full in `app/rails/entitlement.py`.
    """
    if not query or not query.strip():
        return []

    # Guard the caller-supplied k. It reaches a LIMIT, and an unbounded or negative value
    # is either a bug or a way to pull the whole corpus into a 4K context window.
    k = max(1, min(int(k), 10))

    query_vector = embed_texts([query])[0]

    # Imported here rather than at module scope: db/seed.py imports embed_texts from this
    # module while the database is still empty, and app.db builds an engine at import
    # time. Keeping the dependency local means the write path does not pay for the read
    # path.
    from sqlalchemy import text

    from app.db import get_session

    sql = text(
        """
        SELECT chunk_id,
               doc_name,
               chunk_text,
               embedding <=> CAST(:query_vector AS vector) AS distance
        FROM policy_chunks
        WHERE embedding IS NOT NULL
        ORDER BY distance
        LIMIT :k
        """
    )

    with get_session() as session:
        # pgvector's SQLAlchemy type is not in play for a raw text() query, so the vector
        # goes over as its literal string form and is cast server-side.
        rows = session.execute(
            sql, {"query_vector": str(query_vector), "k": k}
        ).mappings().all()

    return [
        {
            "chunk_id": row["chunk_id"],
            "doc_name": row["doc_name"],
            "chunk_text": row["chunk_text"],
            "distance": float(row["distance"]),
        }
        for row in rows
    ]
