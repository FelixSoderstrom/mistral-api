from pathlib import Path
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    # Model Configuration
    MODEL_PATH: str = str(
        Path(__file__).parent / "models" / "dolphin-2.0-mistral-7b.Q5_K_S.gguf"
    )
    MAX_NEW_TOKENS: int = 2048
    DEFAULT_MAX_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95

    # LLaMA.cpp Configuration
    N_GPU_LAYERS: int = 32  # Use all layers for Q5_K_M
    N_BATCH: int = 2048  # Will be adjusted dynamically
    N_THREADS: int = 8  # Will be set to physical CPU core count

    # API Configuration
    API_TITLE: str = "Mistral API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "FastAPI-based API serving Mistral Dolphin GGUF model with dynamic agent instantiation"

    # Rate Limiting
    RATE_LIMIT_SECONDS: int = 60
    RATE_LIMIT_CALLS: int = 10

    # Redis Configuration (for rate limiting)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # CORS Configuration
    CORS_ORIGINS: list = ["http://localhost:3000"]  # Add your React frontend URL

    class Config:
        env_file = ".env"


settings = Settings()
