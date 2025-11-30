import os
from uuid import uuid4
from typing import Optional, List, Dict, Any
from qdrant_client.http import models

from semantic_image_search.backend.config import Config
from semantic_image_search.backend.qdrant_client import QdrantClientManager
from semantic_image_search.backend.embeddings import embed_image_paths
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


class IndexService:
    """
    Handles image indexing operations:
      - Single image indexing
      - Folder-wise batch indexing
      - Auto category inference
      - Batch embedding (efficient)
    """

    def __init__(self):
        log.info("Initializing IndexService...")

        try:
            self.client = QdrantClientManager.get_client()
            QdrantClientManager.ensure_collection()

            self.collection = Config.QDRANT_COLLECTION
            log.info("IndexService initialized successfully", collection=self.collection)

        except Exception as e:
            log.error("Failed to initialize IndexService", error=str(e))
            raise SemanticImageSearchException("Failed to initialize IndexService", e)

    # ---------------------------------------------------------
    # Single Image Index
    # ---------------------------------------------------------
    def index_image(self, image_path: str, category: Optional[str] = None):
        log.info("Indexing single image", image=image_path, category=category)

        try:
            vec = embed_image_paths([image_path])[0]

            payload = {
                "filename": os.path.basename(image_path),
                "path": image_path,
                "category": category,
            }

            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=str(uuid4()),
                        vector=vec,
                        payload=payload,
                    )
                ],
            )

            log.info("Single image indexed successfully", image=image_path)

        except Exception as e:
            log.error("Failed to index single image", image=image_path, error=str(e))
            raise SemanticImageSearchException("Failed to index single image", e)

    # ---------------------------------------------------------
    # Batch (Folder) Index
    # ---------------------------------------------------------
    def index_folder(self, root_folder: str | os.PathLike):
        root_folder = str(root_folder)
        log.info("Starting folder indexing", folder=root_folder)

        exts = (".jpg", ".jpeg", ".png", ".webp")

        for dirpath, _, files in os.walk(root_folder):
            category = os.path.basename(dirpath)

            image_paths: List[str] = []
            payloads: List[Dict[str, Any]] = []

            for f in files:
                if f.lower().endswith(exts):
                    img_path = os.path.join(dirpath, f)
                    image_paths.append(img_path)
                    payloads.append({
                        "filename": os.path.basename(img_path),
                        "path": img_path,
                        "category": category,
                    })

            if not image_paths:
                continue

            log.info("Batch image collection completed",
                     folder=dirpath, total_images=len(image_paths), category=category)

            try:
                vectors = embed_image_paths(image_paths)

                points = [
                    models.PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload=payload,
                    )
                    for vector, payload in zip(vectors, payloads)
                ]

                self.client.upsert(
                    collection_name=self.collection,
                    points=points,
                )

                log.info("Folder indexed successfully",
                         folder=dirpath, total_indexed=len(points))

            except Exception as e:
                log.error("Failed to index folder", folder=dirpath, error=str(e))
                raise SemanticImageSearchException("Failed to index folder", e)

    # ---------------------------------------------------------
    # Clear Collection
    # ---------------------------------------------------------
    def clear_collection(self):
        log.warning("Clearing Qdrant collection", collection=self.collection)

        try:
            self.client.delete(
                collection_name=self.collection,
                filter=models.Filter(must=[]),
            )

            log.info("Collection cleared", collection=self.collection)

        except Exception as e:
            log.error("Failed to clear collection", error=str(e))
            raise SemanticImageSearchException("Failed to clear collection", e)


# CLI TEST
if __name__ == "__main__":
    indexer = IndexService()
