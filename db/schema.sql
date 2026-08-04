-- ============================================================================
-- SentinelBank schema (PROJECT_PLAN.md §1.5)
--
-- Executed by /docker-entrypoint-initdb.d on FIRST BOOT ONLY, against an empty
-- data volume. Schema changes require: docker compose down -v
--
-- Two things in here are load-bearing for the security story and must not be
-- lost in a later edit:
--   1. transactions.description and service_requests.notes are attacker-controlled
--      free text. They are the indirect prompt-injection vectors this whole project
--      exists to demonstrate. Every read path must mark them source=UNTRUSTED.
--   2. audit_log is NEVER queryable by the agent. sql_guard's table allowlist
--      excludes it; this comment is the reminder, the allowlist is the control.
-- ============================================================================

-- Must be the first statement: policy_chunks.embedding depends on this type.
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================================
-- relationship_managers — the entitlement root. Exactly 2 rows when seeded.
-- ============================================================================
CREATE TABLE relationship_managers (
    rm_id       TEXT PRIMARY KEY,   -- 'RM-001', 'RM-002'
    full_name   TEXT NOT NULL,
    branch      TEXT NOT NULL
);


-- ============================================================================
-- customers — 40 rows when seeded (22 -> RM-001, 18 -> RM-002).
--
-- Almost every column here is PII. The entity names in the comments are the
-- Presidio entity types the Phase 3 PII rail detects and redacts; they are the
-- contract between this schema and app/rails/pii.py.
-- ============================================================================
CREATE TABLE customers (
    customer_id     TEXT PRIMARY KEY,   -- 'CUST-0001'

    -- PII: PERSON — reversible pseudonymization, not redaction. Replaced with a
    -- per-request token (Michael Chen -> <PERSON_a3f9>) so the LLM never sees a
    -- real customer name, then rehydrated on output (§3.1.4).
    full_name       TEXT NOT NULL,

    -- PII: US_SSN — Presidio built-in. Format XXX-XX-XXXX. Strategy: redact.
    ssn             TEXT NOT NULL,

    -- PII: US_DRIVER_LICENSE — Presidio built-in. Strategy: redact.
    drivers_license TEXT,

    -- PII: DATE_TIME (date of birth) — Presidio built-in plus a context filter, so
    -- that ordinary transaction dates are not swept up. Strategy: redact.
    date_of_birth   DATE NOT NULL,

    -- PII: EMAIL_ADDRESS — one of the five types LangChain's PIIMiddleware handles
    -- natively. Strategy: redact.
    email           TEXT,

    -- PII: PHONE_NUMBER — Presidio built-in. Strategy: mask.
    phone           TEXT,

    -- PII: LOCATION — Presidio built-in. Strategy: redact.
    street_address  TEXT,
    -- PII: LOCATION — city/state/zip are address components; same entity, same rail.
    city            TEXT,
    state           TEXT,
    zip_code        TEXT,

    risk_rating     TEXT NOT NULL DEFAULT 'LOW'
        CHECK (risk_rating IN ('LOW', 'MEDIUM', 'HIGH')),

    -- Exactly 2 customers are flagged when seeded. Drives the HARD_BLOCK tier:
    -- initiate_wire against an OFAC-flagged customer is refused outright and cannot
    -- be approved by a human (§3.3).
    ofac_flag       BOOLEAN NOT NULL DEFAULT FALSE,

    -- ENTITLEMENT KEY. Every agent query is rewritten to filter on this column,
    -- with rm_id taken from application session state — never from model output or
    -- user text (§2.1). The predicate is injected into the SQL AST, not requested
    -- in the prompt, so the model is structurally incapable of over-fetching.
    rm_id           TEXT NOT NULL REFERENCES relationship_managers (rm_id)
);


-- ============================================================================
-- accounts — 60 rows when seeded (~20 customers hold two).
-- ============================================================================
CREATE TABLE accounts (
    account_id      TEXT PRIMARY KEY,   -- 'ACCT-0001'

    customer_id     TEXT NOT NULL REFERENCES customers (customer_id),

    -- PII: US_BANK_NUMBER — Presidio built-in. 10-12 digits. Strategy: mask,
    -- last 4 visible, because RMs legitimately identify accounts by their last 4.
    account_number  TEXT NOT NULL,

    -- PII: ABA_ROUTING — the project's single CUSTOM Presidio recognizer
    -- (app/rails/recognizers.py). A bare \d{9} pattern floods with false positives
    -- against amounts and IDs, so the recognizer validates the weighted checksum:
    --   (3*(d1+d4+d7) + 7*(d2+d5+d8) + 1*(d3+d6+d9)) mod 10 == 0
    -- db/seed.py generates these with the same formula, so detector and fixtures
    -- agree. Strategy: redact.
    routing_number  TEXT NOT NULL,

    account_type    TEXT NOT NULL
        CHECK (account_type IN ('CHECKING', 'SAVINGS', 'MONEY_MARKET')),
    balance         NUMERIC(14, 2) NOT NULL DEFAULT 0.00,   -- USD
    status          TEXT NOT NULL DEFAULT 'ACTIVE'
        CHECK (status IN ('ACTIVE', 'FROZEN', 'CLOSED')),
    opened_date     DATE NOT NULL
);


-- ============================================================================
-- transactions — 600 rows when seeded, 4 of them carrying injection payloads.
-- ============================================================================
CREATE TABLE transactions (
    txn_id      TEXT PRIMARY KEY,

    account_id  TEXT NOT NULL REFERENCES accounts (account_id),

    amount      NUMERIC(14, 2) NOT NULL,   -- negative = debit

    -- >>> UNTRUSTED — INJECTION VECTOR <<<
    -- Merchant strings are supplied by third-party payees, not by the bank.
    -- Read paths must mark this source=UNTRUSTED (§2.3) so Phase 3 scans and
    -- spotlights it.
    merchant    TEXT,

    -- ========================================================================
    -- >>> UNTRUSTED — PRIMARY INJECTION VECTOR <<<
    --
    -- Free-text memo data controlled by whoever originated the payment. This is
    -- THE attack surface the project demonstrates: 4 of the 6 planted payloads
    -- live in this column, reaching model context through an entirely innocent
    -- RM query about recent transactions.
    --
    -- Mandatory handling (§2.3, §3.2.2, §3.2.3):
    --   * every tool returning this column sets source=UNTRUSTED
    --   * the injection classifier scans it BEFORE it enters model context
    --   * on detection, this field alone is replaced with
    --     [CONTENT QUARANTINED - POSSIBLE INJECTION] and the rest of the row is
    --     kept, so the RM still gets their answer
    --   * surviving content is wrapped in <untrusted_bank_data> delimiters, which
    --     are first stripped from the content itself
    --
    -- Content in this column is data to be reported on. It is never instructions.
    -- ========================================================================
    description TEXT,

    channel     TEXT NOT NULL
        CHECK (channel IN ('ACH', 'WIRE', 'ZELLE', 'POS', 'ATM', 'CHECK')),
    txn_date    TIMESTAMP NOT NULL,
    is_disputed BOOLEAN NOT NULL DEFAULT FALSE
);


-- ============================================================================
-- loans — 18 rows when seeded (12 current, 6 delinquent by 15-120 days).
-- ============================================================================
CREATE TABLE loans (
    loan_id             TEXT PRIMARY KEY,

    customer_id         TEXT NOT NULL REFERENCES customers (customer_id),

    loan_type           TEXT NOT NULL
        CHECK (loan_type IN ('AUTO', 'MORTGAGE', 'PERSONAL', 'HELOC')),
    principal           NUMERIC(14, 2) NOT NULL,
    outstanding_balance NUMERIC(14, 2) NOT NULL,
    monthly_payment     NUMERIC(10, 2) NOT NULL,
    interest_rate       NUMERIC(5, 3) NOT NULL,
    days_delinquent     INTEGER NOT NULL DEFAULT 0
);


-- ============================================================================
-- service_requests — 25 rows when seeded, 2 of them carrying injection payloads.
-- ============================================================================
CREATE TABLE service_requests (
    ticket_id   TEXT PRIMARY KEY,

    customer_id TEXT NOT NULL REFERENCES customers (customer_id),

    category    TEXT NOT NULL
        CHECK (category IN ('DISPUTE', 'CARD_REPLACEMENT', 'FEE_WAIVER', 'ADDRESS_CHANGE')),
    status      TEXT NOT NULL DEFAULT 'OPEN'
        CHECK (status IN ('OPEN', 'PENDING_REVIEW', 'RESOLVED')),

    -- ========================================================================
    -- >>> UNTRUSTED — INJECTION VECTOR <<<
    --
    -- Ticket notes, frequently transcribed straight from what the customer said.
    -- 2 of the 6 planted payloads live here, including payload #6 — a deliberate
    -- delimiter-escape attempt ("</data> New instruction: ... <data>") that exists
    -- to prove the spotlighting rail sanitises its own delimiters before wrapping
    -- (§3.2.3). Same mandatory handling as transactions.description above.
    -- ========================================================================
    notes       TEXT,

    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by  TEXT NOT NULL   -- 'CUSTOMER' or an rm_id such as 'RM-001'
);


-- ============================================================================
-- policy_chunks — ~40-60 chunks from the 6 markdown policy documents.
-- Retrieved content here is bank-authored and therefore TRUSTED.
-- ============================================================================
CREATE TABLE policy_chunks (
    chunk_id    SERIAL PRIMARY KEY,
    doc_name    TEXT NOT NULL,
    chunk_text  TEXT NOT NULL,
    -- 384 dimensions: all-MiniLM-L6-v2. Changing the embedding model means changing
    -- this width, which means down -v and a reseed.
    embedding   vector(384)
);


-- ============================================================================
-- audit_log — tamper-evident, hash-chained (§3.5).
--
-- NEVER QUERYABLE BY THE AGENT. sql_guard's table allowlist deliberately excludes
-- this table; 'SELECT * FROM audit_log' is one of the red-team sql_escalation cases.
--
-- Stores PII-REDACTED text only. The raw query and response never land here.
-- ============================================================================
CREATE TABLE audit_log (
    event_id         SERIAL PRIMARY KEY,
    timestamp        TIMESTAMP NOT NULL DEFAULT NOW(),
    rm_id            TEXT NOT NULL,

    -- PII-redacted before write. Never the raw user text.
    user_query       TEXT,

    -- List of {rail, verdict, score, latency_ms} — feeds the Streamlit rail status
    -- panel and the Phase 4 latency budget table.
    rails_fired      JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_calls       JSONB NOT NULL DEFAULT '[]'::jsonb,

    decision         TEXT NOT NULL
        CHECK (decision IN ('ALLOWED', 'BLOCKED', 'PENDING_APPROVAL', 'APPROVED', 'REJECTED')),

    -- PII-redacted before write.
    response_summary TEXT,

    -- Hash chain: entry_hash = SHA256(prev_hash + canonical_json(payload)).
    -- The genesis entry uses prev_hash = '0' * 64. verify_chain() walks from genesis
    -- and returns the first index where the recomputed hash diverges, which is how
    -- a mutated row is caught.
    prev_hash        TEXT NOT NULL,
    entry_hash       TEXT NOT NULL UNIQUE
);


-- ============================================================================
-- Indexes
-- ============================================================================

-- Entitlement lookups. Every single agent query filters on customers.rm_id, so this
-- is the hottest index in the schema.
CREATE INDEX idx_customers_rm_id ON customers (rm_id);

-- Foreign keys. Postgres does not index the referencing side automatically, and the
-- entitlement predicate reaches these tables by joining back to customers.
CREATE INDEX idx_accounts_customer_id         ON accounts (customer_id);
CREATE INDEX idx_transactions_account_id      ON transactions (account_id);
CREATE INDEX idx_loans_customer_id            ON loans (customer_id);
CREATE INDEX idx_service_requests_customer_id ON service_requests (customer_id);

-- Common non-key filters in RM questions ("recent transactions", "open tickets").
CREATE INDEX idx_transactions_txn_date     ON transactions (txn_date DESC);
CREATE INDEX idx_service_requests_status   ON service_requests (status);

-- Audit chain verification walks the table in insertion order from genesis.
CREATE INDEX idx_audit_log_timestamp ON audit_log (timestamp);

-- Approximate nearest-neighbour index for policy RAG, cosine distance (<=>), which
-- is what pgvector's vector_cosine_ops supports and what app/embeddings.py queries.
--
-- lists = 10 suits the ~40-60 chunk corpus (rule of thumb: rows/1000, floor of 10).
-- Two caveats worth knowing rather than discovering later:
--   * ivfflat builds its centroids from existing rows. This index is created on an
--     empty table at first boot, so after db/seed.py has loaded the chunks, run
--     REINDEX INDEX idx_policy_chunks_embedding; to build meaningful centroids.
--   * at this corpus size the planner will usually pick a sequential scan anyway,
--     and top-3 recall is exact either way. The index is here because the schema
--     should be correct at production scale, not because 50 rows need it.
CREATE INDEX idx_policy_chunks_embedding
    ON policy_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);
