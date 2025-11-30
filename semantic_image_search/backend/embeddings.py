import os
import sys
from typing import List
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from semantic_image_search.backend.config import Config
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


class EmbeddingLoader:
    """
    Loads and manages CLIP embedding models.
    """

    def __init__(self):
        try:
            log.info(
                "Initializing CLIP Embedding Loader",
                model=Config.CLIP_MODEL_NAME,
                checkpoint=Config.CLIP_CHECKPOINT,
                device=Config.DEVICE
            )

            self.embedder = OpenCLIPEmbeddings(
                model_name=Config.CLIP_MODEL_NAME,
                checkpoint=Config.CLIP_CHECKPOINT,
                device=Config.DEVICE
            )

            log.info("CLIP Embedding Model Loaded Successfully")

        except Exception as e:
            log.error(
                "Failed to initialize CLIP embeddings",
                error=str(e),
                model=Config.CLIP_MODEL_NAME,
                checkpoint=Config.CLIP_CHECKPOINT
            )
            raise SemanticImageSearchException("Error loading CLIP Embedding Model", e)

    # ---------------------------------------------------------
    # TEXT → VECTOR
    # ---------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        if not text:
            raise ValueError("Text cannot be empty for embedding")

        log.info("Embedding text", text_preview=text[:40])

        try:
            vec = self.embedder.embed_query(text)
            log.info("Text embedding successful", vector_dim=len(vec))
            return vec

        except Exception as e:
            log.error("Error embedding text", text_preview=text[:40], error=str(e))
            raise SemanticImageSearchException("Failed to embed text", e)

    # ---------------------------------------------------------
    # IMAGE → VECTOR
    # ---------------------------------------------------------
    def embed_image(self, image_path: str) -> List[float]:
        log.info("Embedding single image", image=image_path)

        try:
            vectors = self.embedder.embed_image([image_path])
            vec = vectors[0]

            log.info("Single image embedding successful", vector_dim=len(vec))
            return vec

        except Exception as e:
            log.error("Error embedding image", image=image_path, error=str(e))
            raise SemanticImageSearchException("Failed to embed image", e)

    # ---------------------------------------------------------
    # BATCH IMAGE EMBEDDINGS
    # ---------------------------------------------------------
    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        log.info("Embedding batch images", total_images=len(image_paths))

        try:
            vectors = self.embedder.embed_image(image_paths)
            log.info("Batch image embedding successful", total_images=len(vectors))
            return vectors

        except Exception as e:
            log.error(
                "Error embedding batch images",
                total_images=len(image_paths),
                error=str(e)
            )
            raise SemanticImageSearchException("Failed to embed image batch", e)


# -------------------------------------------------------------
# OPTIONAL: LAZY SINGLETON
# -------------------------------------------------------------
_embedding_loader = None


def get_loader() -> EmbeddingLoader:
    global _embedding_loader
    if _embedding_loader is None:
        _embedding_loader = EmbeddingLoader()
    return _embedding_loader


# Convenience API wrappers
def embed_text(text: str) -> List[float]:
    return get_loader().embed_text(text)


def embed_single_image(image_path: str) -> List[float]:
    return get_loader().embed_image(image_path)


def embed_image_paths(image_paths: List[str]) -> List[List[float]]:
    return get_loader().embed_images(image_paths)
