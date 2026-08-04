"""Application settings loaded from .env via pydantic-settings.

Defaults here mirror .env.example exactly, so the app is runnable without a .env file
and so there is a single place to read what each knob means. See PROJECT_PLAN.md §3.4
for the rail toggles and §1.1 for the hardware constraints behind the LLM settings.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RailsConfig(BaseModel):
    """Per-middleware on/off switches, in §3.4 execution order.

    The core demo of this project is turning rails off and on, so every middleware is
    individually switchable rather than sitting behind one global flag. These values
    are the startup defaults only — POST /config/rails and the Streamlit sidebar
    mutate the live object at runtime.

    Every flag defaults to True: absent or partial configuration must leave the rails
    ON, never off. Fail closed (§6 rule 4).
    """

    # 1. User-input injection scan (before_agent, can jump to end).
    injection_input: bool = True
    # 2. LangChain PIIMiddleware built-ins: email, credit_card.
    pii_builtin: bool = True
    # 3. Presidio through PIIMiddleware's `detector` hook: everything else.
    pii_presidio: bool = True
    # 4. Reversible person pseudonymization; map is per-request, never global.
    person_pseudonym: bool = True
    # 5. HITL risk tiers. Note this does not gate HARD_BLOCK tools — an OFAC-flagged
    #    wire is refused whether or not this rail is enabled.
    hitl: bool = True
    # 6. Tool-result injection scan over UNTRUSTED fields. The signature rail (§3.2.2).
    injection_tool_result: bool = True
    # 7. <untrusted_bank_data> delimiter wrapping, with delimiter sanitisation.
    spotlight: bool = True
    # 8. SHA-256 hash-chained audit write.
    audit: bool = True

    # Mutable at runtime on purpose. The middleware instances baked into the compiled
    # graph hold a reference to *this object* and re-read their flag on every hook, so
    # POST /config/rails and the Streamlit checkboxes take effect on the next message
    # without rebuilding the agent. That is what makes the core demo — flipping a rail
    # off mid-conversation and re-asking the same question — actually demonstrable.
    model_config = {"validate_assignment": True}


class Settings(BaseSettings):
    """Root settings object. Instantiate once — see the `settings` singleton below."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Lets the flat GUARDRAILS_ENABLED__SPOTLIGHT=false form in .env populate the
        # nested RailsConfig model, keeping one commentable line per rail.
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # --- PostgreSQL ---------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "sentinelbank"
    postgres_user: str = "sentinel"
    postgres_password: str = "sentinel"

    # --- Ollama -------------------------------------------------------------
    # OpenAI-compatible endpoint; swapping to vLLM later is this one variable.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "sentinelbank-llm"

    # Never raise this. Defaulting to the model's max context is the most common
    # cause of OOM on a 4 GB card (§1.1 rule 2). The documented fallback direction
    # is 2048, never higher.
    llm_num_ctx: int = 4096

    # Determinism is a hard requirement — the red-team harness asserts against
    # specific responses (§6 rule 3).
    llm_temperature: float = 0.1

    # --- Guardrail models (CPU only; only Ollama touches the GPU) -----------
    # Gated repo; app falls back to protectai/deberta-v3-base-prompt-injection-v2
    # with a loud warning if it cannot be loaded (§1.1 step 5).
    injection_model_id: str = "meta-llama/Prompt-Guard-86M"
    injection_threshold: float = 0.8
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Session ------------------------------------------------------------
    # Owner of the six poisoned demo rows. rm_id always comes from application
    # session state, never from model output or user text (§2.1, §6 rule 5).
    default_rm_id: str = "RM-001"

    # --- Rail toggles -------------------------------------------------------
    guardrails_enabled: RailsConfig = Field(default_factory=RailsConfig)

    @property
    def database_url(self) -> str:
        """SQLAlchemy DSN for the psycopg 3 driver."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Module-level singleton. Import this rather than constructing Settings() again —
# re-reading .env per request is wasted work on an 8 GB machine.
settings = Settings()
