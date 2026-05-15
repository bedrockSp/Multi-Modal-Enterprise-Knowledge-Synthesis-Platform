from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    DATABASE_URL: str
    SECRET_KEY: str
    DATABASE_NAME: str = "bedrock"
    MODE: str = "development"
    API_KEY_1: str
    API_KEY_2: str
    API_KEY_3: str
    API_KEY_4: str
    API_KEY_5: str
    API_KEY_6: str
    OPENAI_API: str
    QUERY_URL: str
    VISION_URL: str
    MAIN_MODEL: str
    VLM_MODEL: str = "qwen3.5:9b"  # Override in .env to match MAIN_MODEL when both should share one Ollama instance
    REMOTE_GPU: bool = False
    USE_VISION_MODEL: bool = True  # VLM runs on every page/slide (PDF, DOCX, PPTX). Set False in .env to disable.
    LOCAL_BASE_URL: str = "http://localhost"

    # YouTube Data API v3 (for tech sensing video enrichment)
    YOUTUBE_API_KEY: str = ""

    # vLLM Configuration (alternative to Ollama for GPU inference)
    USE_VLLM: bool = False  # Set True to use vLLM instead of Ollama
    VLLM_BASE_URL: str = "http://localhost:8010/v1"  # vLLM OpenAI-compat endpoint
    VLLM_MODEL: str = ""  # Model name as loaded in vLLM (e.g., "google/gemma-4-26B-A4B-it")
    VLLM_API_KEY: str = "none"  # vLLM API key (default: no auth)

    # Model token configuration
    MODEL_CONTEXT_TOKENS: int = 128000   # Total context window in tokens
    MODEL_OUTPUT_TOKENS: int = 16000     # Max output tokens per LLM call (num_predict / max_tokens)
    MODEL_OUTPUT_RESERVE: int = 8000     # Reserved output tokens for budget calculations

    # INTERNAL API Configuration (defaults to disabled for backward compatibility)
    INTERNAL_BASE_URL: str = ""
    INTERNAL_CLIENT_KEY: str = ""
    INTERNAL_API_TOKEN: str = ""
    INTERNAL_USER_EMAIL: str = ""
    INTERNAL_MODEL_ID: str = ""
    USE_INTERNAL: bool = False  # Set to True in .env to enable INTERNAL API

    # CPU / Windows mode — device selection for ML components.
    # "auto" picks cuda when torch.cuda.is_available() else cpu. Explicit "cpu" or
    # "cuda" forces the device regardless of detection (useful if a stray CUDA
    # torch wheel was installed on a CPU-only box).
    EMBEDDING_DEVICE: str = "auto"
    CROSS_ENCODER_DEVICE: str = "auto"
    EASYOCR_GPU: bool = False  # True for ~4-7x faster OCR on NVIDIA GPU

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
