#!/usr/bin/env python3
"""
Run RF-DETR fine-tuning natively on macOS (uses MPS / Apple Silicon GPU).

Usage:
    cd label_studio_ml/examples/rfdetr
    python train_local.py

Environment variables (or set in .env):
    LABEL_STUDIO_HOST   - e.g. http://localhost:8080
    LABEL_STUDIO_API_KEY
    PROJECT_ID          - Label Studio project ID to pull annotations from
    TRAINING_EPOCHS     - number of fine-tuning epochs (default: 10)
    MODEL_ROOT          - path to models directory (default: ./models)
"""

import os
import sys
import logging

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Add the rfdetr app directory to path so trainer.py is importable
sys.path.insert(0, os.path.dirname(__file__))

from trainer import prepare_coco_dataset, run_training, CHECKPOINT_CONFIG

MODEL_ROOT = os.getenv("MODEL_ROOT", os.path.join(os.path.dirname(__file__), "models"))
MODEL_FILE = os.getenv("MODEL_FILE", "checkpoint_best_total.pth")
PROJECT_ID = int(os.getenv("PROJECT_ID", "0"))
EPOCHS = int(os.getenv("TRAINING_EPOCHS", "10"))
LS_HOST = os.getenv("LABEL_STUDIO_HOST", "http://localhost:8080")
LS_KEY = os.getenv("LABEL_STUDIO_API_KEY", "")

# Load class names from companion .txt
def load_class_names(model_path):
    txt = os.path.splitext(model_path)[0] + ".txt"
    if not os.path.exists(txt):
        raise FileNotFoundError(f"Class names file not found: {txt}")
    with open(txt) as f:
        return [l.strip() for l in f if l.strip()]

# Download an image using the authenticated SDK client and cache it locally
def make_get_path(value_key, auth_headers):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "rfdetr-train")
    os.makedirs(cache_dir, exist_ok=True)

    def get_path(task):
        task_path = (
            task["data"].get(value_key)
            or task["data"].get("image")
            or next(iter(task["data"].values()), None)
        )
        if task_path is None:
            raise ValueError(f"No image path in task data: {task['data']}")
        if os.path.exists(task_path):
            return task_path

        # Build the full URL if it's a relative path
        url = task_path if task_path.startswith("http") else f"{LS_HOST}{task_path}"

        fname = os.path.basename(url.split("?")[0])
        dest = os.path.join(cache_dir, f"{task.get('id', 0)}__{fname}")
        if os.path.exists(dest):
            return dest

        import requests
        response = requests.get(url, headers=auth_headers)
        response.raise_for_status()
        with open(dest, "wb") as f:
            f.write(response.content)
        return dest

    return get_path


def main():
    import torch
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    if not PROJECT_ID:
        logger.error("Set PROJECT_ID environment variable to your Label Studio project ID")
        sys.exit(1)

    if not LS_KEY:
        logger.error("Set LABEL_STUDIO_API_KEY environment variable")
        sys.exit(1)

    # Fetch annotated tasks
    from label_studio_sdk import LabelStudio
    logger.info(f"Connecting to Label Studio at {LS_HOST}, project {PROJECT_ID}")
    ls = LabelStudio(base_url=LS_HOST, api_key=LS_KEY)
    tasks = [t.model_dump() for t in ls.tasks.list(project=PROJECT_ID, only_annotated=True)]
    logger.info(f"Fetched {len(tasks)} labeled tasks")

    if not tasks:
        logger.error("No labeled tasks found")
        sys.exit(1)

    model_path = os.path.join(MODEL_ROOT, MODEL_FILE)
    class_names = load_class_names(model_path)
    logger.info(f"Loaded {len(class_names)} class names")

    auth_headers = ls._client_wrapper.get_headers()
    get_path = make_get_path("image", auth_headers)

    # Prepare COCO dataset
    dataset_dir = prepare_coco_dataset(
        tasks=tasks,
        class_names=class_names,
        get_path_fn=get_path,
    )

    output_dir = os.path.join(MODEL_ROOT, "training_output")

    # Train (MPS will be picked up automatically by rfdetr via torch device detection)
    import shutil
    best_ckpt = run_training(
        pretrain_weights=model_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        num_classes=180,
        epochs=EPOCHS,
    )

    if best_ckpt:
        shutil.copy(best_ckpt, model_path)
        logger.info(f"New checkpoint saved to {model_path}")
        logger.info("Restart the Docker container to load the updated model:")
        logger.info("  docker-compose down && docker-compose up")
    else:
        logger.error("Training failed — no checkpoint produced")
        sys.exit(1)


if __name__ == "__main__":
    main()
