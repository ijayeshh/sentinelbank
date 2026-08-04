"""SQLAlchemy 2.x ORM models mirroring db/schema.sql exactly (PROJECT_PLAN.md §1.5).

db/schema.sql is authoritative — it is what Postgres actually executes on first boot.
These models must track it; they do not create it. When the two disagree, fix the model.

Column-level security semantics (PII entity types, UNTRUSTED injection vectors) are
documented in schema.sql and repeated here only where a reader of Python alone would
otherwise miss something load-bearing.
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all SentinelBank tables."""


class RelationshipManager(Base):
    """The entitlement root. Exactly 2 rows: RM-001, RM-002."""

    __tablename__ = "relationship_managers"

    rm_id: Mapped[str] = mapped_column(Text, primary_key=True)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    branch: Mapped[str] = mapped_column(Text, nullable=False)

    customers: Mapped[list["Customer"]] = relationship(back_populates="relationship_manager")


class Customer(Base):
    """40 seeded rows: 22 belong to RM-001, 18 to RM-002.

    Nearly every column is PII. See schema.sql for the Presidio entity type mapped to
    each one — that mapping is the contract with app/rails/pii.py.
    """

    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint(
            "risk_rating IN ('LOW', 'MEDIUM', 'HIGH')",
            name="customers_risk_rating_check",
        ),
    )

    customer_id: Mapped[str] = mapped_column(Text, primary_key=True)

    full_name: Mapped[str] = mapped_column(Text, nullable=False)  # PII: PERSON (pseudonymized)
    ssn: Mapped[str] = mapped_column(Text, nullable=False)  # PII: US_SSN
    drivers_license: Mapped[str | None] = mapped_column(Text)  # PII: US_DRIVER_LICENSE
    date_of_birth: Mapped[date] = mapped_column(Date, nullable=False)  # PII: DATE_TIME (DOB)
    email: Mapped[str | None] = mapped_column(Text)  # PII: EMAIL_ADDRESS
    phone: Mapped[str | None] = mapped_column(Text)  # PII: PHONE_NUMBER
    street_address: Mapped[str | None] = mapped_column(Text)  # PII: LOCATION
    city: Mapped[str | None] = mapped_column(Text)  # PII: LOCATION
    state: Mapped[str | None] = mapped_column(Text)  # PII: LOCATION
    zip_code: Mapped[str | None] = mapped_column(Text)  # PII: LOCATION

    risk_rating: Mapped[str] = mapped_column(Text, nullable=False, server_default="LOW")

    # Exactly 2 seeded customers are flagged. Drives the HARD_BLOCK tier: an
    # initiate_wire against one of these is refused and cannot be human-approved.
    ofac_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    # ENTITLEMENT KEY. Queries are rewritten to filter on this at the SQL AST layer,
    # with rm_id taken from session state — never from model output (§2.1).
    rm_id: Mapped[str] = mapped_column(
        Text, ForeignKey("relationship_managers.rm_id"), nullable=False, index=True
    )

    relationship_manager: Mapped["RelationshipManager"] = relationship(back_populates="customers")
    accounts: Mapped[list["Account"]] = relationship(back_populates="customer")
    loans: Mapped[list["Loan"]] = relationship(back_populates="customer")
    service_requests: Mapped[list["ServiceRequest"]] = relationship(back_populates="customer")


class Account(Base):
    """60 seeded rows; roughly 20 customers hold two accounts."""

    __tablename__ = "accounts"
    __table_args__ = (
        CheckConstraint(
            "account_type IN ('CHECKING', 'SAVINGS', 'MONEY_MARKET')",
            name="accounts_account_type_check",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'FROZEN', 'CLOSED')",
            name="accounts_status_check",
        ),
    )

    account_id: Mapped[str] = mapped_column(Text, primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        Text, ForeignKey("customers.customer_id"), nullable=False, index=True
    )

    account_number: Mapped[str] = mapped_column(Text, nullable=False)  # PII: US_BANK_NUMBER
    # PII: ABA_ROUTING — the project's one custom recognizer, checksum-validated.
    routing_number: Mapped[str] = mapped_column(Text, nullable=False)

    account_type: Mapped[str] = mapped_column(Text, nullable=False)
    balance: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False, server_default="0.00")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="ACTIVE")
    opened_date: Mapped[date] = mapped_column(Date, nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="accounts")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="account")


class Transaction(Base):
    """600 seeded rows, 4 of which carry planted injection payloads in `description`.

    `merchant` and `description` are attacker-controlled free text. Any tool returning
    them must mark the payload source=UNTRUSTED so the Phase 3 rails scan and spotlight
    them before they reach model context (§2.3, §3.2.2).
    """

    __tablename__ = "transactions"
    __table_args__ = (
        CheckConstraint(
            "channel IN ('ACH', 'WIRE', 'ZELLE', 'POS', 'ATM', 'CHECK')",
            name="transactions_channel_check",
        ),
    )

    txn_id: Mapped[str] = mapped_column(Text, primary_key=True)
    account_id: Mapped[str] = mapped_column(
        Text, ForeignKey("accounts.account_id"), nullable=False, index=True
    )

    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)  # negative = debit

    # UNTRUSTED — supplied by third-party payees.
    merchant: Mapped[str | None] = mapped_column(Text)

    # UNTRUSTED — PRIMARY INJECTION VECTOR. Free-text memo controlled by whoever
    # originated the payment; 4 of the 6 planted payloads live here. Content is data
    # to be reported on, never instructions. Quarantined field-by-field on detection.
    description: Mapped[str | None] = mapped_column(Text)

    channel: Mapped[str] = mapped_column(Text, nullable=False)
    txn_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    is_disputed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    account: Mapped["Account"] = relationship(back_populates="transactions")


class Loan(Base):
    """18 seeded rows: 12 current, 6 delinquent between 15 and 120 days."""

    __tablename__ = "loans"
    __table_args__ = (
        CheckConstraint(
            "loan_type IN ('AUTO', 'MORTGAGE', 'PERSONAL', 'HELOC')",
            name="loans_loan_type_check",
        ),
    )

    loan_id: Mapped[str] = mapped_column(Text, primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        Text, ForeignKey("customers.customer_id"), nullable=False, index=True
    )

    loan_type: Mapped[str] = mapped_column(Text, nullable=False)
    principal: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    outstanding_balance: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    monthly_payment: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    interest_rate: Mapped[Decimal] = mapped_column(Numeric(5, 3), nullable=False)
    days_delinquent: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    customer: Mapped["Customer"] = relationship(back_populates="loans")


class ServiceRequest(Base):
    """25 seeded rows, 2 of which carry planted injection payloads in `notes`.

    `notes` is UNTRUSTED — often transcribed straight from what the customer said.
    Payload #6 lives here and attempts a delimiter escape against spotlighting.
    """

    __tablename__ = "service_requests"
    __table_args__ = (
        CheckConstraint(
            "category IN ('DISPUTE', 'CARD_REPLACEMENT', 'FEE_WAIVER', 'ADDRESS_CHANGE')",
            name="service_requests_category_check",
        ),
        CheckConstraint(
            "status IN ('OPEN', 'PENDING_REVIEW', 'RESOLVED')",
            name="service_requests_status_check",
        ),
    )

    ticket_id: Mapped[str] = mapped_column(Text, primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        Text, ForeignKey("customers.customer_id"), nullable=False, index=True
    )

    category: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="OPEN", index=True)

    # UNTRUSTED — INJECTION VECTOR. Same handling as transactions.description.
    notes: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    created_by: Mapped[str] = mapped_column(Text, nullable=False)  # 'CUSTOMER' or an rm_id

    customer: Mapped["Customer"] = relationship(back_populates="service_requests")


class PolicyChunk(Base):
    """~40-60 chunks from the 6 markdown policy documents. Bank-authored, so TRUSTED."""

    __tablename__ = "policy_chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_name: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    # 384 dimensions: all-MiniLM-L6-v2. Queried with cosine distance (<=>).
    embedding: Mapped[Any | None] = mapped_column(Vector(384))


class AuditLog(Base):
    """Tamper-evident hash-chained request log (§3.5).

    NEVER agent-queryable: sql_guard's table allowlist excludes `audit_log`, and
    'SELECT * FROM audit_log' is an explicit red-team case. Stores PII-redacted text
    only — the raw query and response never reach this table.
    """

    __tablename__ = "audit_log"
    __table_args__ = (
        CheckConstraint(
            "decision IN ('ALLOWED', 'BLOCKED', 'PENDING_APPROVAL', 'APPROVED', 'REJECTED')",
            name="audit_log_decision_check",
        ),
    )

    event_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )
    rm_id: Mapped[str] = mapped_column(Text, nullable=False)

    user_query: Mapped[str | None] = mapped_column(Text)  # PII-redacted before write

    # List of {rail, verdict, score, latency_ms}.
    rails_fired: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )
    tool_calls: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )

    decision: Mapped[str] = mapped_column(Text, nullable=False)
    response_summary: Mapped[str | None] = mapped_column(Text)  # PII-redacted before write

    # entry_hash = SHA256(prev_hash + canonical_json(payload)); genesis prev_hash is
    # '0' * 64. UNIQUE on entry_hash makes a duplicated chain link a write error
    # rather than a silent fork.
    prev_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
