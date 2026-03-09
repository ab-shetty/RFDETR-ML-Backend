import logging

from PIL import Image
from control_models.base import ControlModel
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)


class RFDETRRectangleLabelsModel(ControlModel):
    """RF-DETR control model for RectangleLabels + perRegion Taxonomy tasks."""

    type = "RectangleLabels"
    model_path = "checkpoint_best_total.pth"

    @classmethod
    def is_control_matched(cls, control) -> bool:
        if not control.to_name:
            return False
        if control.objects[0].tag != "Image":
            return False
        return control.tag == cls.type

    def _get_box_label(self, class_id: int) -> Optional[str]:
        """Return the LS rectangle label for a predicted class_id."""
        if not self.class_names or class_id >= len(self.class_names):
            return None
        class_name = self.class_names[class_id]
        # Use label_map if populated (class name matched to LS label)
        if self.label_map:
            return self.label_map.get(class_name)
        # Fallback: single-label config (e.g. "Product")
        available = list(self.control.labels_attrs.keys())
        return available[0] if len(available) == 1 else None

    def _get_taxonomy_path(self, class_id: int) -> Optional[List[str]]:
        """Look up the taxonomy path for a predicted class_id."""
        if not self.taxonomy_path_map or not self.class_names:
            return None
        if class_id >= len(self.class_names):
            return None
        class_name = self.class_names[class_id]
        path = self.taxonomy_path_map.get(class_name)
        if path is None:
            logger.debug(f"No taxonomy path for class '{class_name}' (id={class_id})")
        return path

    def predict_regions(self, path) -> List[Dict]:
        image = Image.open(path).convert("RGB")
        width, height = image.size

        detections = self.model.predict(image, threshold=self.model_score_threshold)

        regions = []
        for i in range(len(detections.xyxy)):
            score = float(detections.confidence[i])
            class_id = int(detections.class_id[i])
            x1, y1, x2, y2 = detections.xyxy[i].tolist()

            box_label = self._get_box_label(class_id)
            if box_label is None:
                logger.debug(f"No label for class_id={class_id}, skipping")
                continue

            region_id = self.make_region_id()

            # Bounding box region
            rect_region = {
                "id": region_id,
                "from_name": self.from_name,
                "to_name": self.to_name,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [box_label],
                    "x": (x1 / width) * 100,
                    "y": (y1 / height) * 100,
                    "width": ((x2 - x1) / width) * 100,
                    "height": ((y2 - y1) / height) * 100,
                },
                "score": score,
            }
            regions.append(rect_region)

            # Taxonomy region linked to the bounding box
            tax_path = self._get_taxonomy_path(class_id)
            if tax_path and self.taxonomy_from_name:
                class_name = self.class_names[class_id]
                logger.debug(
                    f"Detection: class='{class_name}' -> taxonomy={tax_path}, "
                    f"score={score:.3f}, bbox=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]"
                )
                tax_region = {
                    "id": self.make_region_id(),
                    "from_name": self.taxonomy_from_name,
                    "to_name": self.taxonomy_to_name or self.to_name,
                    "type": "taxonomy",
                    "parentID": region_id,
                    "value": {
                        "taxonomy": [tax_path],
                    },
                    "score": score,
                }
                regions.append(tax_region)

        return regions
