"""Runtime settings for the AI Dataset Cleaner.

Everything is driven by environment variables so the same code runs locally
(Ollama + SQLite) and hosted (Groq/Gemini + Postgres).  Copy `.env.example`
to `.env` for local development; set the same variables in the Render
dashboard for hosting.
"""
import os
from dataclasses import dataclass, asdict, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load a local .env file if present (no python-dotenv dependency needed).
_env_path = os.path.join(ROOT, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _fh:
        for _line in _fh:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _v = _v.split(" #", 1)[0].strip().strip('"').strip("'")
                os.environ.setdefault(_k.strip(), _v)


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# Default model per provider (all have free tiers except openai/anthropic).
PROVIDER_DEFAULTS = {
    "groq":   {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.5-flash"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct:free"},
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "anthropic": {"base_url": "https://api.anthropic.com/v1", "model": "claude-3-5-haiku-latest"},
    "ollama": {"base_url": os.environ.get("OLLAMA_URL", "http://localhost:11434"), "model": "llama3.2:3b"},
}


@dataclass
class Settings:
    # ---- AI provider -------------------------------------------------------------
    # groq | gemini | openrouter | openai | anthropic | ollama
    provider: str = os.environ.get("LLM_PROVIDER", "ollama").lower()
    api_key: str = os.environ.get("LLM_API_KEY", "")
    model: str = os.environ.get("LLM_MODEL", "")          # empty -> provider default
    base_url: str = os.environ.get("LLM_BASE_URL", "")    # empty -> provider default
    temperature: float = float(os.environ.get("LLM_TEMPERATURE", "0.1"))
    num_ctx: int = _int("OLLAMA_NUM_CTX", 8192)           # Ollama only
    request_timeout: int = _int("LLM_TIMEOUT", 180)
    # 1 = one model call per column (deepest). Hosted free tiers have daily token
    # limits, so 3 columns per call is a good default there.
    columns_per_call: int = _int("COLUMNS_PER_CALL", 1 if os.environ.get("LLM_PROVIDER", "ollama") == "ollama" else 3)
    include_clean_columns: bool = _bool("INCLUDE_CLEAN_COLUMNS", False)
    summarize_plan: bool = _bool("SUMMARIZE_PLAN", True)
    allow_rule_fallback: bool = _bool("ALLOW_RULE_FALLBACK", True)
    # Let visitors change provider/key from the Settings panel? Keep False when hosted.
    allow_client_settings: bool = _bool("ALLOW_CLIENT_SETTINGS", os.environ.get("LLM_PROVIDER", "ollama") == "ollama")

    # ---- storage -------------------------------------------------------------------
    # postgresql://...  (Neon/Render/Supabase) or sqlite:///./data/app.db
    database_url: str = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(ROOT, 'data', 'app.db')}")
    upload_dir: str = field(default_factory=lambda: os.path.join(ROOT, "data", "uploads"))
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 10)
    max_rows: int = _int("MAX_ROWS", 100_000)
    max_columns: int = _int("MAX_COLUMNS", 200)
    dataset_ttl_days: int = _int("DATASET_TTL_DAYS", 7)
    max_datasets_per_session: int = _int("MAX_DATASETS_PER_SESSION", 20)
    max_concurrent_jobs: int = _int("MAX_CONCURRENT_JOBS", 2)

    # ---- web -----------------------------------------------------------------------
    # Comma-separated origins allowed to call the API (your GitHub Pages URL). "*" = any.
    allowed_origins: str = os.environ.get("ALLOWED_ORIGINS", "*")
    serve_frontend: bool = _bool("SERVE_FRONTEND", True)

    def resolved_model(self) -> str:
        return self.model or PROVIDER_DEFAULTS.get(self.provider, {}).get("model", "")

    def resolved_base_url(self) -> str:
        return (self.base_url or PROVIDER_DEFAULTS.get(self.provider, {}).get("base_url", "")).rstrip("/")

    def to_dict(self, public: bool = True):
        d = asdict(self)
        d["model"] = self.resolved_model()
        d["base_url"] = self.resolved_base_url()
        if public:
            d["api_key"] = "set" if self.api_key else ""
            d.pop("database_url", None)
            d.pop("upload_dir", None)
        return d

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                current = getattr(self, k)
                if isinstance(current, bool):
                    v = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int):
                    v = int(v)
                elif isinstance(current, float):
                    v = float(v)
                elif isinstance(current, str):
                    v = str(v).strip()
                setattr(self, k, v)
        return self


settings = Settings()
os.makedirs(settings.upload_dir, exist_ok=True)
if settings.provider == "ollama" and not settings.api_key:
    settings.api_key = ""
