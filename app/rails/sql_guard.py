"""Deterministic SQL safety via sqlglot AST parsing — no regex, no LLM.

SELECT-only, single statement, table allowlist (audit_log is never agent-queryable),
no comments or `;`, LIMIT injected at 20 and clamped at 50. Returns SqlVerdict.
Unparseable input is rejected: fail closed, never open (PROJECT_PLAN.md §2.2).

--------------------------------------------------------------------------------
THREAT MODEL — read this before changing anything below
--------------------------------------------------------------------------------
This module is the boundary between LLM-authored text and the customer database.
Everything upstream of it (the system prompt, the tool description, the model's own
"judgement") is advisory. This is the part that is *structurally* incapable of
letting a dangerous statement through, and it is the only part a reviewer should
have to trust.

Two adversaries produce the SQL that arrives here:

1. An attacker who controls free text inside the database itself. The demo plants
   payloads in `transactions.description` and `service_requests.notes`; an innocent
   RM query retrieves them, they land in the model's context, and the model may then
   emit SQL the attacker chose ("dump every ssn", "read the audit log"). The model is
   3B and follows instructions imperfectly — assume it *will* be steered sometimes.

2. The model itself, being wrong. Hallucinated table names, unbounded result sets
   that blow the 4K context window, joins that fan out to the whole portfolio.

Design rules that follow, and that must survive any future edit:

- **Decisions come from the AST, never from scanning the string for keywords.**
  A keyword scan is defeated by casing, comments, whitespace, string literals, and
  encoding tricks (`SEL/**/ECT`, `"drop"` as a column alias, `DrOp`). The parser has
  already resolved all of that by the time we inspect nodes. The only place we touch
  raw text is the tokenizer pass in step 2, and even there we read *token types*, not
  substrings, so a `;` inside a string literal is correctly seen as data.
- **Fail closed.** Every branch that cannot prove a query is safe rejects it. That
  includes unparseable input, unrecognised node types, and a LIMIT whose value we
  cannot evaluate at validation time.
- **Allowlist, never blocklist, for both statement shape and tables.** A blocklist is
  a bet that we enumerated every dangerous thing; Postgres has more statement types
  than anyone remembers, and sqlglot grows new ones between releases.
- **Distinct reason strings.** Phase 4's red-team harness (§4.2) asserts on *which*
  rail fired and *why*, so these codes are a contract, not debug output. Changing one
  breaks tests deliberately — that is the point.

The LIMIT clamp is a safety control as much as a cost control: a 4096-token context
(§1.1 rule 2) cannot absorb a thousand-row result, and truncation mid-result is
exactly the condition under which a small model starts inventing figures.

Order of operations for the wider pipeline is fixed by §2.2 and lives in
`app/rails/entitlement.py`: validate here -> inject the rm_id predicate -> re-validate
the final string here again. This module is deliberately free of any dependency on
entitlement, timing, config, or the database, so it can be unit-tested in isolation
with no services running.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.tokens import TokenType

# Postgres everywhere. Parsing with the wrong dialect would silently mis-model
# constructs (casts, quoting, `::`) and could accept something the server reads
# differently — a validator that does not agree with the executor is not a validator.
DIALECT = "postgres"

# --------------------------------------------------------------------------------
# Allowlists and limits
# --------------------------------------------------------------------------------

# The seven tables the agent is permitted to read (§2.2). Anything else — including
# tables that exist and are harmless — is rejected, because "harmless" is a judgement
# call and this rail does not make judgement calls.
ALLOWED_TABLES = frozenset(
    {
        "customers",
        "accounts",
        "transactions",
        "loans",
        "service_requests",
        "relationship_managers",
        "policy_chunks",
    }
)

# `audit_log` is the tamper-evident record of what the agent did (§3.5). An agent that
# can read its own audit trail can be steered into summarising, leaking, or reasoning
# about other RMs' activity, and it is the first thing an attacker probes after a
# successful injection. It gets its own reason code rather than falling through to the
# generic allowlist rejection, so the red-team harness can prove *this specific* rule
# fired and did not merely coincide with the allowlist.
FORBIDDEN_TABLES = frozenset({"audit_log"})

# Postgres system catalogs. Reading these leaks the schema, the role list, running
# queries, and file paths — the reconnaissance step before a real escalation attempt.
SYSTEM_SCHEMAS = frozenset({"information_schema", "pg_catalog"})
SYSTEM_PREFIX = "pg_"

# The only schema qualifier an allowlisted table may carry. A qualified name pointing
# anywhere else is either reconnaissance or a mistake; both are rejected.
DEFAULT_SCHEMA = "public"

# §2.2 step 6. DEFAULT is what an unbounded query becomes; MAX is the ceiling a query
# that asked for more is clamped to.
DEFAULT_LIMIT = 20
MAX_LIMIT = 50

# Statement/DDL/DCL node classes that must never appear anywhere in the tree — not at
# the root, not inside a CTE, not inside a subquery. The root-type check in step 5
# already rejects these when they are the whole statement; this list exists because
# sqlglot will happily parse a mutation *nested* inside something that looks like a
# SELECT, and because `exp.Command` is sqlglot's fallback for statements it does not
# model natively (REVOKE, CALL, EXECUTE, and anything added to Postgres after this
# sqlglot release). Rejecting `Command` is what makes the SELECT-only rule airtight
# against syntax we have never seen.
FORBIDDEN_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Grant,
    exp.Copy,
    exp.Command,  # REVOKE / CALL / EXECUTE / unmodelled statements
    exp.Set,  # SET ROLE, SET search_path — session tampering
    exp.Use,
    exp.Refresh,
    exp.Attach,
    exp.Detach,
    exp.Pragma,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

# The only root shapes a read-only rail accepts: a plain SELECT, or a set operation
# (UNION / INTERSECT / EXCEPT) whose branches are themselves validated by the same
# whole-tree walk below.
ALLOWED_ROOTS: tuple[type[exp.Expression], ...] = (exp.Select, exp.SetOperation)


@dataclass(frozen=True)
class SqlVerdict:
    """The structured result required by §2.2 step 7.

    Frozen because a verdict is evidence: it is written to the audit log and asserted
    on by the red-team harness, and no caller should be able to edit one after the
    fact.

    allowed:  whether the statement may reach the database.
    reason:   "ok" when allowed; otherwise a stable, specific rejection code. Some
              codes carry a `:detail` suffix (the offending node class or table name)
              so an operator reading the audit log knows what actually happened.
    safe_sql: the *regenerated* SQL when allowed — never the caller's original string,
              so downstream code cannot accidentally execute un-normalised text —
              and "" when rejected.
    """

    allowed: bool
    reason: str
    safe_sql: str


def _reject(reason: str) -> SqlVerdict:
    """Every rejection path funnels through here, so `safe_sql` is always empty."""
    return SqlVerdict(allowed=False, reason=reason, safe_sql="")


def validate_sql(sql: str) -> SqlVerdict:
    """Validate LLM-authored SQL and return a verdict with clamped, regenerated SQL.

    The steps run in the order given by §2.2 and each one returns immediately on
    failure. The ordering is deliberate: cheap structural checks that need no parse
    tree run first, and the checks whose reason codes are most diagnostic (stacked
    statements, comments) run before the broader ones, so the audit log records the
    most specific explanation available rather than a generic parse failure.
    """
    # --- 0. Nothing to validate -------------------------------------------------
    # An empty tool argument means the model produced no SQL. Fail closed rather than
    # letting an empty string flow onward to be concatenated into something else.
    if not sql or not sql.strip():
        return _reject("empty_sql")

    # --- 1/2. Tokenizer pass: semicolons and comments ---------------------------
    # Both checks happen at the token level rather than with `";" in sql` or a regex,
    # because the tokenizer already knows the difference between a separator and the
    # same character inside a string literal. `WHERE description = 'paid; refunded'`
    # is legitimate RM data and must not be rejected, while `SELECT 1; DROP ...` must.
    #
    # Comments are rejected outright (§2.2 step 5) for two reasons: they are the
    # classic vehicle for smuggling text past naive filters and for terminating the
    # rest of a statement (`... -- ` swallowing an appended predicate), and a machine
    # -generated query has no legitimate need for them. Rejecting them also protects
    # the entitlement rail, whose injected `rm_id` predicate must never be commentable
    # out of existence.
    try:
        tokens = sqlglot.tokenize(sql, dialect=DIALECT)
    except SqlglotError:
        return _reject("unparseable_sql")

    for token in tokens:
        if token.token_type == TokenType.SEMICOLON:
            return _reject("semicolon_separator")
        if token.comments:
            return _reject("sql_comment")

    # --- 3. Exactly one statement ----------------------------------------------
    # Blocks stacked queries: the payload shape where a benign SELECT is followed by a
    # mutation. Step 1 already rejects the `;` such a payload needs, so this is a
    # second, independent line of defence rather than the only one — if a future
    # sqlglot learns to split statements some other way, this still catches it.
    # `parse` can yield None entries for empty trailing fragments ("SELECT 1;;"); they
    # are not statements and must not be counted as such.
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except SqlglotError:
        return _reject("unparseable_sql")

    if not statements:
        return _reject("unparseable_sql")
    if len(statements) > 1:
        return _reject("stacked_statements")

    # --- 4. Parse ---------------------------------------------------------------
    # Unparseable input is rejected, never passed through on the theory that Postgres
    # will reject it anyway. If we cannot model a statement, we cannot reason about
    # it, and a validator that forwards what it does not understand is decorative.
    try:
        tree = sqlglot.parse_one(sql, dialect=DIALECT)
    except SqlglotError:
        return _reject("unparseable_sql")

    if tree is None:
        return _reject("unparseable_sql")

    # --- 5. Root must be a SELECT (or a set operation over SELECTs) -------------
    # Allowlist, not blocklist: we name the two shapes we accept, and every one of the
    # dozens of other node classes sqlglot can produce falls into the rejection. The
    # class name is appended to the reason so the audit log says `:Drop` rather than
    # leaving an operator to guess.
    if not isinstance(tree, ALLOWED_ROOTS):
        return _reject(f"non_select_statement:{type(tree).__name__}")

    # --- 6. No forbidden node anywhere in the tree ------------------------------
    # The root check above only inspects the outermost node. A mutation can hide one
    # level down — inside a CTE (`WITH x AS (DELETE ... RETURNING *) SELECT * FROM x`,
    # which Postgres genuinely supports), inside a subquery, or inside a set-operation
    # branch. This walks the entire tree, which is the only way to be sure.
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return _reject(f"forbidden_statement_node:{type(node).__name__}")

    # --- 7. Table allowlist over the whole tree ---------------------------------
    # `find_all(exp.Table)` descends into CTE bodies, subqueries, joins, and set
    # operations, so a disallowed table cannot hide behind nesting.
    #
    # CTE aliases surface here as Table nodes too (`WITH x AS (...) SELECT * FROM x`
    # yields a table named `x`), but they are references to a query we have already
    # validated, not to a physical relation. They are excluded from the allowlist
    # check — while the CTE's *body* is still walked, so a CTE that reads `audit_log`
    # is caught by the audit_log rule below and not laundered by the alias.
    cte_aliases = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    physical_tables = []
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = (table.db or "").lower()
        catalog = (table.catalog or "").lower()

        # System catalogs: schema-qualified (`information_schema.tables`,
        # `pg_catalog.pg_tables`) or bare (`pg_stat_activity`, which resolves through
        # the default search_path). Checked before the CTE-alias exclusion so a CTE
        # cannot be named `pg_class` to slip past.
        if schema in SYSTEM_SCHEMAS or catalog in SYSTEM_SCHEMAS:
            return _reject("system_catalog_access")
        if name.startswith(SYSTEM_PREFIX):
            return _reject("system_catalog_access")

        if name in cte_aliases and not schema:
            continue

        # audit_log gets its own code — see FORBIDDEN_TABLES above.
        if name in FORBIDDEN_TABLES:
            return _reject("audit_log_access")

        if name not in ALLOWED_TABLES:
            return _reject(f"table_not_allowlisted:{name}")

        # An allowlisted table reached through an unexpected schema is not the table
        # we allowlisted. Only unqualified names and explicit `public.` are accepted.
        if schema and schema != DEFAULT_SCHEMA:
            return _reject("system_catalog_access")
        if catalog:
            return _reject("system_catalog_access")

        physical_tables.append(name)

    # A statement that reads no table at all (`SELECT 1`, `SELECT version()`) is not
    # something this agent has any reason to emit, and it is the shape probing queries
    # take. Fail closed.
    if not physical_tables:
        return _reject("no_table_reference")

    # --- 8. No system functions -------------------------------------------------
    # Defence in depth, slightly beyond the literal wording of §2.2, which names the
    # `pg_*` catalogs but not the `pg_*` functions that reach the same data by another
    # route: `pg_read_file`, `pg_ls_dir`, `pg_sleep` (a trivial denial of service on a
    # single-box demo), `pg_stat_file`. `exp.Anonymous` is sqlglot's node for any
    # function it does not model natively, which is where all of these land; the
    # prefix test runs against the parsed function name, not the raw string.
    for func in tree.find_all(exp.Anonymous):
        if func.name.lower().startswith(SYSTEM_PREFIX):
            return _reject("system_function_call")

    # --- 9. LIMIT enforcement, on the AST ---------------------------------------
    # Done by mutating the tree and re-generating, never by appending " LIMIT 20" to a
    # string. String concatenation is wrong here in a way that matters: it silently
    # produces invalid SQL for set operations and for queries with trailing clauses,
    # and any construct that can change what an appended clause binds to is an
    # injection primitive.
    limit = tree.args.get("limit")

    if limit is None:
        tree.set("limit", exp.Limit(expression=exp.Literal.number(DEFAULT_LIMIT)))
    else:
        value = limit.expression

        # The bound must be a plain integer literal we can evaluate right now. A
        # parameter, a placeholder, an arithmetic expression, or a subquery means the
        # true row count is decided at execution time, after this rail has finished —
        # so we cannot prove the ceiling holds. Fail closed rather than guess.
        if not isinstance(value, exp.Literal) or value.is_string:
            return _reject("unresolvable_limit")

        try:
            requested = int(value.this)
        except (TypeError, ValueError):
            return _reject("unresolvable_limit")

        # A negative or zero LIMIT is not a Postgres error for 0 but is meaningless
        # from the agent and suggests the value was assembled rather than intended.
        if requested < 1:
            return _reject("unresolvable_limit")

        if requested > MAX_LIMIT:
            limit.set("expression", exp.Literal.number(MAX_LIMIT))

    # --- 10. Regenerate ---------------------------------------------------------
    # The caller receives sqlglot's rendering of the validated tree, not the original
    # text. Anything the parser did not preserve — stray whitespace, unusual quoting,
    # anything that survived tokenization without becoming a node — is discarded here
    # rather than forwarded to the database.
    try:
        safe_sql = tree.sql(dialect=DIALECT)
    except SqlglotError:
        return _reject("unparseable_sql")

    return SqlVerdict(allowed=True, reason="ok", safe_sql=safe_sql)
