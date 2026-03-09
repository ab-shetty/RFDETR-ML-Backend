"""RF-DETR fine-tuning utilities for Label Studio annotations."""

import json
import logging
import os
import random
import shutil
import tempfile

from PIL import Image

logger = logging.getLogger(__name__)

# Architecture params that must match the checkpoint
CHECKPOINT_CONFIG = dict(
    resolution=384,
    patch_size=16,
    positional_encoding_size=24,
    num_windows=2,
    dec_layers=2,
    out_feature_indexes=[3, 6, 9, 12],
)


def prepare_coco_dataset(tasks, class_names, get_path_fn, val_split=0.2):
    """
    Convert Label Studio rectangle-label annotations to a COCO-format dataset.

    Returns path to a temp directory with the structure:
        <tmpdir>/train/_annotations.coco.json + images
        <tmpdir>/valid/_annotations.coco.json + images
    """
    categories = [
        {"id": i, "name": name, "supercategory": ""}
        for i, name in enumerate(class_names)
    ]
    class_to_id = {name: i for i, name in enumerate(class_names)}

    # Collect tasks that have at least one non-cancelled annotation with results
    annotated = []
    for task in tasks:
        anns = [
            a for a in task.get("annotations", [])
            if not a.get("skipped") and not a.get("was_cancelled") and a.get("result")
        ]
        if anns:
            annotated.append((task, anns[0]))  # use first annotation

    if not annotated:
        raise ValueError("No annotated tasks found for training")

    logger.info(f"Preparing COCO dataset from {len(annotated)} annotated tasks")

    random.shuffle(annotated)
    split_idx = max(1, int(len(annotated) * (1 - val_split)))
    # Ensure at least 1 task in val; if dataset is tiny, use first task for both
    train_items = annotated[:split_idx]
    val_items = annotated[split_idx:] or annotated[:max(1, split_idx // 4)]

    tmpdir = tempfile.mkdtemp(prefix="rfdetr_train_")
    logger.info(f"Training dataset dir: {tmpdir}")

    def write_split(items, split_name):
        out_dir = os.path.join(tmpdir, split_name)
        os.makedirs(out_dir, exist_ok=True)

        images = []
        annotations = []
        ann_id = 0

        for img_id, (task, annotation) in enumerate(items):
            # Download / resolve image path
            try:
                img_path = get_path_fn(task)
            except Exception as e:
                logger.warning(f"Skipping task {task.get('id')}: {e}")
                continue

            ext = os.path.splitext(img_path)[1] or ".jpg"
            dest_name = f"img_{img_id}{ext}"
            dest_path = os.path.join(out_dir, dest_name)
            shutil.copy(img_path, dest_path)

            with Image.open(dest_path) as img:
                w, h = img.size

            images.append({"id": img_id, "file_name": dest_name, "width": w, "height": h})

            for result in annotation.get("result", []):
                if result.get("type") != "rectanglelabels":
                    continue
                val = result.get("value", {})
                labels = val.get("rectanglelabels", [])
                if not labels:
                    continue
                label = labels[0]
                cat_id = class_to_id.get(label)
                if cat_id is None:
                    logger.debug(f"Label '{label}' not in class_names, skipping")
                    continue

                # Convert percent → pixels (COCO bbox: [x, y, w, h])
                x = val["x"] * w / 100
                y = val["y"] * h / 100
                bw = val["width"] * w / 100
                bh = val["height"] * h / 100

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                })
                ann_id += 1

        coco_json = {"images": images, "annotations": annotations, "categories": categories}
        with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
            json.dump(coco_json, f)

        logger.info(f"[{split_name}] {len(images)} images, {len(annotations)} bbox annotations")

    write_split(train_items, "train")
    write_split(val_items, "valid")

    return tmpdir


def run_training(pretrain_weights, dataset_dir, output_dir, num_classes, epochs=10, **train_kwargs):
    """
    Fine-tune an RFDETRBase model.

    Returns path to the best checkpoint produced, or None if training failed.
    """
    from rfdetr import RFDETRBase

    logger.info(f"Starting RF-DETR fine-tuning: epochs={epochs}, dataset={dataset_dir}")
    os.makedirs(output_dir, exist_ok=True)

    model = RFDETRBase(
        pretrain_weights=pretrain_weights,
        num_classes=num_classes,
        **CHECKPOINT_CONFIG,
    )

    model.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=epochs,
        dataset_file="roboflow",
        tensorboard=False,
        run_test=False,
        progress_bar=True,
        num_workers=0,  # avoid shared memory issues in Docker
        **train_kwargs,
    )

    best_ckpt = os.path.join(output_dir, "checkpoint_best_total.pth")
    if os.path.exists(best_ckpt):
        logger.info(f"Best checkpoint saved at: {best_ckpt}")
        return best_ckpt

    logger.warning("No best checkpoint found after training")
    return None
