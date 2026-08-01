import logging

from PIL import Image
from control_models.base import CASCADE_ENABLED, CASCADE_FLOOR, SHELF_TAGS_ENABLED, ControlModel
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

    def predict_regions(self, path, cascade_mode: Optional[str] = None) -> List[Dict]:
        """
        :param cascade_mode: "off" | "cascade" | "cascade_shelf_tags", or None
            to use this process's CASCADE_ENABLED/SHELF_TAGS_ENABLED env vars
            (the default for every caller except the "Retrieve Predictions"
            UI action, which lets a user override it per run).
        """
        image = Image.open(path).convert("RGB")
        width, height = image.size

        if cascade_mode is None:
            cascade_enabled = CASCADE_ENABLED
            shelf_tags_enabled = SHELF_TAGS_ENABLED
        else:
            cascade_enabled = cascade_mode in ("cascade", "cascade_shelf_tags")
            shelf_tags_enabled = cascade_mode == "cascade_shelf_tags"

        # Call the model with a low floor, then apply the real per-class cutoff
        # below — filtering at the model call would throw away class-specific
        # detail before we can use it. With the cascade on, drop the floor to
        # CASCADE_FLOOR so sub-threshold detections reach the cascade, which can
        # promote the real ones (recall recovery).
        predict_floor = min(self.min_prediction_threshold(), CASCADE_FLOOR) if cascade_enabled else self.min_prediction_threshold()
        detections = self.model.predict(image, threshold=predict_floor)

        regions = []
        for i in range(len(detections.xyxy)):
            score = float(detections.confidence[i])
            class_id = int(detections.class_id[i])
            x1, y1, x2, y2 = detections.xyxy[i].tolist()

            box_label = self._get_box_label(class_id)
            if box_label is None:
                logger.debug(f"No label for class_id={class_id}, skipping")
                continue

            class_name = self.class_names[class_id] if class_id < len(self.class_names) else None
            effective_threshold = self.get_effective_threshold(class_name) if class_name else self.model_score_threshold

            if cascade_enabled and class_name:
                # The cascade governs the whole [CASCADE_FLOOR, inf) range: it can
                # reject above-threshold false positives AND promote below-threshold
                # real detections. So don't pre-drop on the per-class threshold here —
                # pass the threshold in and let the cascade tier the decision.
                from cascade.embedding_match import get_backbone_nn_module
                from cascade.pipeline import Decision, verify_detection

                crop = image.crop((x1, y1, x2, y2))
                decision = verify_detection(
                    crop=crop,
                    class_name=class_name,
                    detector_confidence=score,
                    effective_threshold=effective_threshold,
                    expected_text=self.expected_text,
                    reference_gallery=self.reference_gallery,
                    nn_model=get_backbone_nn_module(self.model),
                    cascade_floor=CASCADE_FLOOR,
                )
                if decision == Decision.AUTO_REJECT:
                    logger.debug(f"Cascade rejected '{class_name}' score={score:.3f}")
                    continue
            elif score < effective_threshold:
                # No cascade: plain per-class threshold filter.
                logger.debug(f"Dropping '{class_name}' score={score:.3f} < threshold={effective_threshold:.3f}")
                continue

            regions.extend(self._emit_regions(class_id, box_label, x1, y1, x2, y2, width, height, score))

        if shelf_tags_enabled and self.tag_class_map:
            self._apply_tag_corrections(image, regions)

        return regions

    def _emit_regions(self, class_id, box_label, x1, y1, x2, y2, width, height, score) -> List[Dict]:
        """Build the rectangle region (+ linked taxonomy region) for one detection."""
        region_id = self.make_region_id()
        out = [{
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
        }]
        tax_path = self._get_taxonomy_path(class_id)
        if tax_path and self.taxonomy_from_name:
            # A per-region classification is linked to its region by REUSING the
            # region's id, not by a separate id + parentID (parentID in Label
            # Studio means region nesting -- a child region in the outliner
            # tree -- which is a different feature). On deserialize LS groups
            # results by id into one Area, so the box and its SKU must share it.
            out.append({
                "id": region_id,
                "from_name": self.taxonomy_from_name,
                "to_name": self.taxonomy_to_name or self.to_name,
                "type": "taxonomy",
                "value": {"taxonomy": [tax_path]},
                "score": score,
            })
        return out

    def _apply_tag_corrections(self, image, regions) -> None:
        """Correct the SKU (taxonomy) on each box using the shelf tag in its
        column. RF-DETR/the cascade localize the box; the shelf tag — the
        store's own label — names the product, which is the harder part and
        where the detector is weakest.

        Measured on held-out frames this raised class-aware precision (and
        recall on test) with no extra boxes, unlike proposing new boxes on top
        of the cascade, which mostly added false positives (see the PR history).
        Proposing new boxes from tags (propose_from_tags in cascade/shelf_tags)
        is kept as a utility for the low-recall regime where the cascade is off.
        """
        from cascade.shelf_tags import detect_tags, lookup_class

        tags = detect_tags(image)
        if not tags:
            return
        name_to_id = {n: i for i, n in enumerate(self.class_names)}
        # taxonomy results share their box's id (see _emit_regions)
        tax_by_region = {r["id"]: r for r in regions if r.get("type") == "taxonomy"}

        for r in regions:
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            cx = (v["x"] + v["width"] / 2) / 100.0
            cy = (v["y"] + v["height"] / 2) / 100.0
            # tag sits just below the box center, same column
            near = [t for t in tags if abs(t["x"] - cx) < 0.09 and 0 < (t["y"] - cy) < 0.13]
            if not near:
                continue
            tag = min(near, key=lambda t: t["y"] - cy)
            cls = lookup_class(tag["name"], self.tag_class_map)
            if not cls or cls not in name_to_id:
                continue
            tax_path = self._get_taxonomy_path(name_to_id[cls])
            if not tax_path:
                continue
            existing = tax_by_region.get(r["id"])
            if existing:
                if existing["value"].get("taxonomy") != [tax_path]:
                    logger.debug(f"Shelf-tag corrected SKU to '{cls}' (tag '{tag['name']}')")
                    existing["value"]["taxonomy"] = [tax_path]
            elif self.taxonomy_from_name:
                regions.append({
                    "id": r["id"],
                    "from_name": self.taxonomy_from_name,
                    "to_name": self.taxonomy_to_name or self.to_name,
                    "type": "taxonomy",
                    "value": {"taxonomy": [tax_path]},
                    "score": r.get("score", 0.5),
                })
