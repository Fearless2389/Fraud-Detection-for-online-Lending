"""Application configuration.

Every value the application needs at runtime is resolved here, from the
environment, exactly once. Nothing else in the codebase reads ``os.environ``
directly and no credential appears in source.

Two consequences worth knowing when you read the rest of the code:

* There are no hardcoded hosts, ports or origins. Moving this service from a
  laptop to a cloud host is a change of environment values, not of code.
* The fraud cost figures (``cost_fp_inr`` / ``cost_fn_inr``) live here rather
  than in the model code. They are *business* inputs — what the bank loses by
  wrongly blocking a genuine customer versus what it loses by approving a
  fraudster — and the decision threshold is derived from them. Treating them
  as configuration is deliberate: a risk owner should be able to change the
  bank's risk appetite without a model retrain or a code deploy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Repository root: backend/app/core/config.py -> up three levels.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- application ----------------------------------------------------
    app_env: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    app_name: str = "Aegis Fraud Detection API"
    api_version: str = "v1"

    # --- security -------------------------------------------------------
    # Clients must send this in the X-API-Key header. There is no default:
    # a missing key must fail loudly at startup rather than silently leave
    # the scoring endpoint open.
    api_key: str = Field(min_length=8)

    # Comma-separated in the environment, a list here.
    #
    # `NoDecode` is required: pydantic-settings tries to JSON-decode any
    # complex-typed field coming from a .env file *before* validators run, so
    # a plain `CORS_ORIGINS=http://localhost:5173` would fail to parse as JSON
    # and raise at startup. NoDecode hands the raw string to the validator
    # below instead, which is what lets the .env stay human-readable.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )

    # --- persistence ----------------------------------------------------
    database_url: str = ""

    # --- AI layer -------------------------------------------------------
    # Gemini narrates explanations that the deterministic models have already
    # produced. It never contributes to an approve/block decision. Disabling it
    # degrades explanation prose to templates; it does not change any outcome.
    gemini_api_key: str = ""
    # Pinned rather than using the `gemini-flash-latest` alias: a moving alias
    # can change model behaviour between rehearsal and demo. Verified live on
    # 2026-08-20; earlier 2.x flash names have been retired server-side.
    gemini_model: str = "gemini-3.6-flash"
    gemini_enabled: bool = True

    # --- filesystem -----------------------------------------------------
    # Kept outside OneDrive-synced folders: syncing a ~1 GB dataset or a venv
    # causes file locks mid-write and burns cloud quota for zero benefit.
    data_dir: Path = Path("C:/dev/data/baf")
    artifact_dir: Path = PROJECT_ROOT / "backend" / "ml" / "artifacts"

    # --- decision economics ---------------------------------------------
    # The approve/review/block thresholds are NOT configured. They are derived
    # from these four business inputs by app.services.policy.derive_policy,
    # then capped by analyst capacity. Changing the bank's risk appetite is a
    # change to these numbers, not to a threshold someone tuned by hand.
    cost_fp_inr: float = Field(default=1_500.0, gt=0)
    cost_fn_inr: float = Field(default=45_000.0, gt=0)
    cost_review_inr: float = Field(default=200.0, ge=0)
    analyst_catch_rate: float = Field(default=0.90, gt=0.0, le=1.0)

    # Share of application volume the fraud operations team can manually
    # review. Bayes-optimal thresholds assume unlimited review capacity;
    # this is the constraint that makes the policy staffable.
    max_review_rate: float = Field(default=0.05, ge=0.0, le=1.0)

    # --- similarity search ------------------------------------------------
    # Minimum leaf-overlap before an application is shown as "resembling
    # confirmed fraud". Chosen by measurement, not by feel: on 4,000 held-out
    # applications against a 400-case index, top-1 similarity separates as
    #
    #     threshold   genuine firing   fraud firing   lift
    #        0.45         32.3%           75.5%       2.3x
    #        0.55          5.3%           30.6%       5.8x
    #        0.60          1.3%           10.2%       7.9x   <- selected
    #        0.65          0.3%            2.0%       6.2x
    #
    # 0.60 peaks the lift and fires on 1.3% of genuine applications. A lower
    # cut-off fires on a third of all traffic at barely 2x lift, which is
    # alert fatigue rather than a signal - an analyst who sees the panel on
    # every third case stops reading it.
    similarity_min_score: float = Field(default=0.60, ge=0.0, le=1.0)
    similarity_moderate_score: float = Field(default=0.55, ge=0.0, le=1.0)
    similarity_seed_size: int = Field(default=400, ge=0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept ``a,b,c`` from the environment as a list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("data_dir", "artifact_dir", mode="after")
    @classmethod
    def _resolve_against_project_root(cls, path: Path) -> Path:
        """Anchor relative paths to the repository, not the working directory.

        A relative ARTIFACT_DIR such as `./backend/ml/artifacts` resolves
        differently depending on where the process was started - uvicorn run
        from `backend/` looks for `backend/backend/ml/artifacts` and silently
        starts with scoring disabled. Anchoring to the project root makes the
        service behave identically however it is launched.
        """
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @field_validator("cost_review_inr")
    @classmethod
    def _review_cheaper_than_false_block(cls, review_cost: float, info) -> float:
        """A review that costs more than a wrongful block is never worth doing.

        Caught at startup rather than producing a silently collapsed review
        band at scoring time.
        """
        cost_fp = info.data.get("cost_fp_inr")
        if cost_fp is not None and review_cost >= cost_fp:
            raise ValueError(
                f"cost_review_inr ({review_cost}) must be below cost_fp_inr ({cost_fp}); "
                "otherwise manual review is never cheaper than blocking"
            )
        return review_cost

    @property
    def gemini_available(self) -> bool:
        """True only when narration is both enabled and actually configured."""
        return self.gemini_enabled and bool(self.gemini_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that the .env file is parsed once. FastAPI routes depend on this
    via ``Depends(get_settings)``, which also makes it trivial to override in
    tests.
    """
    return Settings()
