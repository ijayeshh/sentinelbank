"""SQLAlchemy 2.x engine and session factory for the on-premise Postgres + pgvector DB.

One engine per process, created at import time. Connection settings come from
app/config.py, which reads .env (PROJECT_PLAN.md §2 stack, §1.4).

Pool sizing is deliberately small: this is a single-user demo on an 8GB machine where
Docker is capped at 3GB, so a large pool would only reserve memory that Ollama needs.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

# psycopg 3 driver, DSN assembled in Settings.database_url.
engine = create_engine(
    settings.database_url,
    # Verify a connection is alive before handing it out. Docker Desktop drops idle
    # connections when WSL 2 suspends, which otherwise surfaces as a stale-connection
    # error on the first query after the machine wakes.
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=5,
    # SQL echo is off: statements can contain customer PII, and this project keeps
    # PII out of logs by construction. Use the audit log (redacted) instead.
    echo=False,
    future=True,
)

# expire_on_commit=False so tool functions can return ORM objects that stay readable
# after their session closes — the rails layer inspects those rows after commit.
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


@contextmanager
def get_session() -> Iterator[Session]:
    """Session scope for tools and scripts: commits on success, rolls back on error.

    Read tools do not strictly need the commit, but write tools (file_dispute,
    waive_fee, ...) run through this same scope, and a rollback on exception is what
    keeps a failed HITL-approved mutation from leaving a half-applied change.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    with get_session() as session:
        yield session
