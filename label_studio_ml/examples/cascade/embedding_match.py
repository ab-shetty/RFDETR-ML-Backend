"""Embedding-similarity verification signal for the pre-annotation cascade.

Reuses the backbone feature-extraction approach already written (and used
for diversity sampling) in the team's Colab notebooks: hook the second-to-
last DINOv2 windowed-attention layer, mean-pool over windows and tokens to
get a 384-dim embedding. Here it's used the other way — comparing a single
detection crop's embedding against a per-class reference gallery built from
already-labeled crops, to catch "confident but wrong class" detections that
a confidence threshold alone can't.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

_TRANSFORM = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_HIDDEN_DIM = 384
_NUM_WINDOWS_SQ = 4  # 2x2 windowed attention


def get_backbone_nn_module(rfdetr_model):
    """Unwrap an RFDETRBase/RFDETRNano instance down to the raw nn.Module.

    RFDETRBase(...).model is a ModelContext; .model.model is the LWDETR
    nn.Module that actually has .backbone. Verified against the loaded
    checkpoint_best_total.pth in this repo — if RF-DETR's wrapper structure
    changes, this is the one place that needs updating.
    """
    return rfdetr_model.model.model


def extract_embedding(nn_model, image: Image.Image, device=None) -> np.ndarray:
    """Embed a single crop. Returns a (384,) vector, or zeros if the hook
    doesn't fire (e.g. RF-DETR internals changed shape) — callers should
    treat an all-zero embedding as "no signal" rather than a real match.
    """
    if device is None:
        device = next(nn_model.parameters()).device

    features_store = {}

    def hook_fn(module, inp, output):
        features_store["backbone"] = output[0]

    hook = nn_model.backbone[0].encoder.encoder.encoder.layer[11].register_forward_hook(hook_fn)
    nn_model.eval()
    try:
        tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            try:
                nn_model(tensor)
            except Exception:
                pass  # hook fires before any error further down the detection head
        if "backbone" not in features_store:
            logger.warning("embedding_match: backbone hook did not fire, returning zero vector")
            return np.zeros(_HIDDEN_DIM, dtype=np.float32)
        out = features_store["backbone"]  # [1*num_windows, tokens, hidden_dim]
        out = out.view(1, _NUM_WINDOWS_SQ, -1, _HIDDEN_DIM)
        pooled = out.mean(dim=[1, 2])  # [1, hidden_dim]
        return pooled.cpu().float().numpy()[0]
    finally:
        hook.remove()


def load_reference_gallery(path: str) -> Dict[str, np.ndarray]:
    """Load {class_name: centroid_embedding} from a .npz built by
    scripts/build_reference_gallery.py. Returns {} if the file doesn't exist
    — callers should treat that as "no embedding signal available" rather
    than an error, same convention as load_class_thresholds().
    """
    if not os.path.exists(path):
        logger.info(f"No reference gallery found at {path} — embedding-match signal disabled.")
        return {}
    data = np.load(path, allow_pickle=False)
    names = data["class_names"]
    centroids = data["centroids"]
    return {str(name): centroids[i] for i, name in enumerate(names)}


def nearest_classes(embedding: np.ndarray, gallery: Dict[str, np.ndarray], k: int = 3) -> List[Tuple[str, float]]:
    """Cosine-similarity-ranked nearest classes in the gallery, closest first."""
    if not gallery or not np.any(embedding):
        return []
    names = list(gallery.keys())
    matrix = np.stack([gallery[n] for n in names])
    emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    sims = mat_norm @ emb_norm
    order = np.argsort(-sims)[:k]
    return [(names[i], float(sims[i])) for i in order]


def embedding_agrees(
    embedding: np.ndarray,
    gallery: Dict[str, np.ndarray],
    predicted_class: str,
    top_k: int = 3,
) -> Optional[bool]:
    """Does the predicted class fall within the embedding's top-k nearest
    gallery classes? Returns None (no signal) if the class isn't in the
    gallery at all or the gallery is empty.
    """
    if predicted_class not in gallery:
        return None
    top = nearest_classes(embedding, gallery, k=top_k)
    if not top:
        return None
    return predicted_class in {name for name, _ in top}
