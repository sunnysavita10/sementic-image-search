import shutil
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse

from semantic_image_search.backend.config import Config
from semantic_image_search.backend.query_translator import translate_query
from semantic_image_search.backend.ingestion import IndexService
from semantic_image_search.backend.retriever import ImageSearchService
from semantic_image_search.backend.logger import GLOBAL_LOGGER as log
from semantic_image_search.backend.exception.custom_exception import SemanticImageSearchException


app = FastAPI(
    title="Semantic Image Search API",
    description="CLIP + Qdrant + LLM Query Translator",
    version="1.0",
)

# Lazy singletons
search_service = None
index_service = None


@app.on_event("startup")
def init_services():
    global search_service, index_service
    search_service = ImageSearchService()
    index_service = IndexService()
    log.info("Services initialized successfully")


# ---------------------------------------------------------
# INGEST ENDPOINT
# ---------------------------------------------------------
@app.post("/ingest")
def ingest_images(
    folder_path: Optional[str] = Query(None, description="Folder of images to index"),
):
    folder = folder_path or str(Config.IMAGES_ROOT)
    log.info("Ingest request received", folder=folder)

    try:
        index_service.index_folder(folder)
        log.info("Ingestion completed", folder=folder)
        return {"message": f"Indexed images from {folder}"}

    except Exception as e:
        log.error("Ingestion failed", folder=folder, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# TRANSLATE ENDPOINT
# ---------------------------------------------------------
@app.get("/translate")
def translate(q: str):
    log.info("Translate request received", query=q)

    try:
        translated = translate_query(q)
        log.info("Query translated", original=q, translated=translated)
        return {"input": q, "translated": translated}

    except Exception as e:
        log.error("Translation failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# TEXT SEARCH ENDPOINT
# ---------------------------------------------------------
@app.get("/search-text")
def search_text_endpoint(
    q: str,
    k: int = 5,
    category: Optional[str] = None,
    save_results: bool = False,
):
    log.info("Text search request received", query=q, top_k=k, category=category)

    try:
        translated = translate_query(q)
        log.info("Query translated for text search", translated=translated)

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_text(translated, k=k, metadata_filter=metadata_filter)

        log.info("Text search completed", total_results=len(results.points))

        resp = [
            {
                "filename": p.payload.get("filename"),
                "path": p.payload.get("path"),
                "category": p.payload.get("category"),
                "score": p.score,
            }
            for p in results.points
        ]

        folder = None
        if save_results and results.points:
            folder = search_service.save_results(results)
            log.info("Search results saved locally", folder=folder)

        return {"query": q, "translated": translated, "k": k, "saved_folder": folder, "results": resp}

    except Exception as e:
        log.error("Text search failed", query=q, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})


# ---------------------------------------------------------
# IMAGE SEARCH ENDPOINT
# ---------------------------------------------------------
@app.post("/search-image")
def search_image_endpoint(
    file: UploadFile = File(...),
    k: int = 5,
    category: Optional[str] = None,
    save_results: bool = False,
):
    log.info("Image search request received", filename=file.filename)

    try:
        if not file.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"error": "Only image files allowed"})

        Config.QUERY_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
        query_path = Config.QUERY_IMAGE_ROOT / file.filename

        with query_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        log.info("Uploaded query image saved", path=str(query_path))

        metadata_filter = {"category": category} if category else None

        results = search_service.search_by_image(str(query_path), k=k, metadata_filter=metadata_filter)

        resp = [
            {
                "filename": p.payload.get("filename"),
                "path": p.payload.get("path"),
                "category": p.payload.get("category"),
                "score": p.score,
            }
            for p in results.points
        ]

        folder = None
        if save_results and results.points:
            folder = search_service.save_results(results)
            log.info("Search results saved locally", folder=folder)

        return {"query_image": str(query_path), "k": k, "saved_folder": folder, "results": resp}

    except Exception as e:
        log.error("Image search failed", filename=file.filename, error=str(e))
        return JSONResponse(status_code=500, content={"error": str(e), "type": type(e).__name__})
