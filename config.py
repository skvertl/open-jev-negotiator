"""Application configuration and environment settings."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_URL: str = "http://localhost:11434/v1"
    LLM_API_KEY: str = "dummy"
    LLM_MODEL: str = "qwen2.5:7b"
    OLLAMA_URL: str = "http://localhost:11434"
    JEV_BACKEND: str = "ollama"  # "ollama", "http" (Open-Jev), or "mock"
    JEV_URL: str = "http://127.0.0.1:8791/v1/systemone"
    JEV_MODEL_PATH: str = "ZefanCai/Open-Jev-9B"
    MOCK_MODE: bool = False

    DEFAULT_COMPANY_VALUATION: int = 2_000_000
    DEFAULT_INVESTOR_BUDGET: int = 700_000
    DEFAULT_INVESTOR_MAX_EQUITY: float = 30.0
    DEFAULT_INVESTOR_INTEREST: float = 0.70
    DEFAULT_INVESTOR_PATIENCE: int = 4


settings = Settings()
