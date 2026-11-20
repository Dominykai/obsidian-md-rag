
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent.parent
CHROMA_DIR = ROOT_DIR / "rag" / "data" / "chroma"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True
    )

    GEMINI_WEB_1PSID: str
    GEMINI_WEB_1PSIDTS: str 

settings = Settings()