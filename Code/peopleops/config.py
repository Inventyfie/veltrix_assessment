from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


def _path(variable: str, default: str) -> Path:
    return Path(os.getenv(variable, default)).expanduser().resolve()


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path
    manifest_path: Path
    db_path: Path
    index_path: Path
    today: date
    provider: str
    model: str
    api_key: str
    request_timeout: float
    provider_timeout: float


def load_settings() -> Settings:
    today_text = os.getenv("APP_TODAY", "").strip()
    today = date.fromisoformat(today_text) if today_text else date.today()
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider != "openai":
        raise ValueError("LLM_PROVIDER must be openai; other providers are not implemented")
    return Settings(
        corpus_dir=_path("CORPUS_DIR", "./data/corpus"),
        manifest_path=_path("MANIFEST_PATH", "./data/manifest.json"),
        db_path=_path("DB_PATH", "./data/veltrix.db"),
        index_path=_path("INDEX_PATH", "./data/.peopleops-index.sqlite3"),
        today=today,
        provider=provider,
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "27")),
        provider_timeout=float(os.getenv("PROVIDER_TIMEOUT_SECONDS", "12")),
    )
