"""Settings read from the environment in one place."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Limits:
    """Hard caps for one agent run and for the whole day."""

    model_calls_per_run: int = field(default_factory=lambda: _int("LIMIT_MODEL_CALLS", 12))
    tool_calls_per_run: int = field(default_factory=lambda: _int("LIMIT_TOOL_CALLS", 20))
    tokens_per_run: int = field(default_factory=lambda: _int("LIMIT_TOKENS", 150_000))
    seconds_per_run: int = field(default_factory=lambda: _int("LIMIT_SECONDS", 120))
    provider_calls_per_day: int = field(default_factory=lambda: _int("LIMIT_DAILY_CALLS", 400))


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Keys stay server side and are never logged."""

    model: str = field(
        default_factory=lambda: os.getenv("MODEL", "google_genai:gemini-3.1-flash-lite")
    )
    fallback_model: str | None = field(default_factory=lambda: os.getenv("FALLBACK_MODEL"))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT / "data")))
    timezone: str = field(default_factory=lambda: os.getenv("TIMEZONE", "Europe/London"))
    gate_samples: int = field(default_factory=lambda: _int("GATE_SAMPLES", 3))
    gate_temperature: float = field(
        default_factory=lambda: float(os.getenv("GATE_TEMPERATURE", "0.7"))
    )
    cors_origin: str = field(
        default_factory=lambda: os.getenv("CORS_ORIGIN", "http://localhost:3000")
    )
    limits: Limits = field(default_factory=Limits)


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
