import os

# Root folder name
ROOT = "semantic_image_search"

# Folder & file structure definition
structure = {
    "src": {
        "api": {
            "routes": {
                "search.py": "",
                "indexer.py": "",
                "upload.py": "",
                "health.py": "",
            },
            "schemas": {
                "search_request.py": "",
                "search_response.py": "",
                "index_request.py": ""
            },
            "middleware": {
                "auth_middleware.py": "",
                "rate_limit_middleware.py": ""
            },
            "main.py": "",
            "dependencies.py": "",
        },
        "core": {
            "config.py": "",
            "logger.py": "",
            "constants.py": "",
            "exceptions.py": "",
        },
        "services": {
            "image_service.py": "",
            "embedding_service.py": "",
            "vectorstore_service.py": "",
            "caption_service.py": "",
            "query_rewriter.py": "",
            "reranker_service.py": "",
            "preprocessing_service.py": "",
        },
        "models": {
            "clip_model.py": "",
            "blip_model.py": "",
            "sentence_transformer.py": "",
            "git_model.py": "",
            "model_loader.py": "",
        },
        "storage": {
            "local_storage.py": "",
            "s3_storage.py": "",
            "gcs_storage.py": "",
            "utils_image.py": "",
        },
        "vectorstore": {
            "qdrant_store.py": "",
            "faiss_store.py": "",
            "metadata_store.py": "",
        },
        "tasks": {
            "index_task.py": "",
            "cleanup_task.py": "",
            "workers.py": "",
        },
        "monitoring": {
            "metrics.py": "",
            "tracing.py": "",
            "sentry_logging.py": "",
        },
        "utils": {
            "time_utils.py": "",
            "json_utils.py": "",
            "file_utils.py": "",
            "validation_utils.py": "",
        }
    },
    "notebooks": {
        "experiments.ipynb": "",
        "evaluate_recall_at_k.ipynb": "",
        "model_comparison.ipynb": "",
    },
    "tests": {
        "test_api.py": "",
        "test_vectorstore.py": "",
        "test_embeddings.py": "",
        "test_captioning.py": "",
        "test_storage.py": "",
    },
    "config": {
        "settings.yaml": "",
        "logging.yaml": "",
        "qdrant.yaml": "",
        "s3.yaml": "",
    },
    "infra": {
        "k8s": {
            "deployment.yaml": "",
            "service.yaml": "",
            "configmap.yaml": "",
            "secrets.yaml": "",
            "hpa.yaml": "",
        },
        "github-actions": {
            "ci.yml": "",
            "cd.yml": "",
        },
        "nginx": {
            "nginx.conf": "",
        },
        "Dockerfile": "",
        "docker-compose.yml": "",
    },
    "requirements.txt": "",
    "README.md": "",
}

# Function to recursively create folders + files
def create_structure(base_path, struct):
    for name, content in struct.items():
        path = os.path.join(base_path, name)

        # If content is a dict = it's a folder
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            # content is empty → create empty file
            os.makedirs(base_path, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)


# Create everything
create_structure(ROOT, structure)

print("\n🎉 Folder structure created successfully!")
print(f"📁 Root Directory: {ROOT}")
