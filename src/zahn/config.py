from __future__ import annotations

import socket
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout: int = 60

    poll_interval: int = 5
    max_attempts: int = 3

    # Auto-generate from hostname:pid if not set
    worker_id: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.worker_id:
            object.__setattr__(
                self, "worker_id", f"{socket.gethostname()}:{os.getpid()}"
            )


def load_settings() -> Settings:
    return Settings()
