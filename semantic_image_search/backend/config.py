import os
from pathlib import Path
from dotenv import load_dotenv
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


# ------------------------------------------------------------
# 1) Resolve BASE_DIR
# ------------------------------------------------------------
try:
    BASE_DIR = Path(__file__).resolve().parents[2]
    log.info("BASE_DIR resolved successfully", base_dir=str(BASE_DIR))
except Exception as e:
    log.error("Failed to resolve BASE_DIR", error=str(e))
    raise


# ------------------------------------------------------------
# 2) Load .env File
# ------------------------------------------------------------
env_path = BASE_DIR / ".env"

if env_path.exists():
    load_dotenv(env_path)
    log.info(".env loaded successfully", env_path=str(env_path))
else:
    log.warning(".env file not found", env_path=str(env_path))


class Config:
    BASE_DIR: Path = BASE_DIR

    # ------------------- PATHS -------------------
    IMAGES_ROOT: Path = Path(os.getenv("IMAGES_ROOT", BASE_DIR / "images"))
    log.info("IMAGES_ROOT configured", value=str(IMAGES_ROOT))

    QUERY_IMAGE_ROOT: Path = Path(os.getenv("QUERY_IMAGE_ROOT", BASE_DIR / "data/query_images"))
    log.info("QUERY_IMAGE_ROOT configured", value=str(QUERY_IMAGE_ROOT))

    RETRIEVED_ROOT: Path = Path(os.getenv("RETRIEVED_ROOT", BASE_DIR / "data/retrieved"))
    log.info("RETRIEVED_ROOT configured", value=str(RETRIEVED_ROOT))

    # ------------------- CLIP ---------------------
    CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
    log.info("CLIP_MODEL_NAME loaded", value=CLIP_MODEL_NAME)

    CLIP_CHECKPOINT: str = os.getenv("CLIP_CHECKPOINT", "laion2b_s34b_b79k")
    log.info("CLIP_CHECKPOINT loaded", value=CLIP_CHECKPOINT)

    DEVICE: str = os.getenv("DEVICE", "cpu")
    log.info("DEVICE selected", value=DEVICE)

    # ------------------- QDRANT -------------------
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    if QDRANT_URL:
        log.info("QDRANT_URL loaded", value=QDRANT_URL)
    else:
        log.warning("QDRANT_URL missing in environment")

    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    if QDRANT_API_KEY:
        log.info("QDRANT_API_KEY loaded (hidden)")
    else:
        log.warning("QDRANT_API_KEY missing in environment")

    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "semantic-image-search")
    log.info("QDRANT_COLLECTION loaded", value=QDRANT_COLLECTION)

    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", 512))
    log.info("VECTOR_SIZE loaded", value=VECTOR_SIZE)

    # ------------------- OPENAI -------------------
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    log.info("OPENAI_MODEL loaded", value=OPENAI_MODEL)

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        log.warning("OPENAI_API_KEY missing")


# ------------------------------------------------------------
# 7) Final global config success log
# ------------------------------------------------------------
config = Config()
log.info("Config initialized successfully", status="OK")
