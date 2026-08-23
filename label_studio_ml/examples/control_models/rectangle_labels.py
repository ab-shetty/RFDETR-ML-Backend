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

    def predict_regions(
        self, path, cascade_mode: Optional[str] = None, detection_floor: Optional[float] = None,
        propose_boxes: Optional[bool] = None, name_proposals: Optional[bool] = None,
        name_boxes: Optional[bool] = None,
    ) -> List[Dict]:
        """
        :param cascade_mode: "off" | "cascade" | "cascade_shelf_tags", or None
            to use this process's CASCADE_ENABLED/SHELF_TAGS_ENABLED env vars
            (the default for every caller except the "Retrieve Predictions"
            UI action, which lets a user override it per run).
        :param name_boxes: name every box by showing the drawn boxes to a vision
            model (see cascade/box_naming.py), overriding the class head's
            guesses. None uses BOX_NAMING_ENABLED. This is the naming stage for
            a store the detector has never seen: the head learned its names from
            one visit, the shelf tag is reprinted whenever the shelf changes.

        :param detection_floor: the confidence below which nothing is kept, for
            this call only. None uses the configured defaults.

            This is deliberately ONE knob across all three modes, because
            "discard below this confidence" means the same thing in each -- and
            two separate cutoffs were genuinely confusing to use. With the
            cascade on it replaces CASCADE_FLOOR (how deep the cascade may reach
            to promote a detection) and the per-class thresholds still set the
            confident/uncertain boundary above it. With the cascade off there is
            no two-tier system, so it is simply the flat cutoff, overriding the
            per-class thresholds.

            Raising it always yields fewer boxes and lowering it always yields
            more, in every mode. That was not true when the UI exposed the
            threshold separately: below CASCADE_FLOOR the floor silently won, so
            0.10 and 0.20 gave identical results with the cascade on.
        """
        image = Image.open(path).convert("RGB")
        width, height = image.size

        if cascade_mode is None:
            cascade_enabled = CASCADE_ENABLED
            shelf_tags_enabled = SHELF_TAGS_ENABLED
        else:
            cascade_enabled = cascade_mode in ("cascade", "cascade_shelf_tags")
            shelf_tags_enabled = cascade_mode == "cascade_shelf_tags"

        cascade_floor = detection_floor if detection_floor is not None else CASCADE_FLOOR

        # Call the model with a low floor, then apply the real per-class cutoff
        # below — filtering at the model call would throw away class-specific
        # detail before we can use it. With the cascade on, drop the floor to
        # cascade_floor so sub-threshold detections reach the cascade, which can
        # promote the real ones (recall recovery).
        if cascade_enabled:
            predict_floor = min(self.min_prediction_threshold(), cascade_floor)
        else:
            predict_floor = detection_floor if detection_floor is not None else self.min_prediction_threshold()
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
            if cascade_enabled:
                # Per-class thresholds keep their meaning here: they are the
                # confident/uncertain boundary, not the cutoff. The cutoff is
                # cascade_floor.
                effective_threshold = (
                    self.get_effective_threshold(class_name) if class_name else self.model_score_threshold
                )
            elif detection_floor is not None:
                effective_threshold = detection_floor
            elif class_name:
                effective_threshold = self.get_effective_threshold(class_name)
            else:
                effective_threshold = self.model_score_threshold

            if cascade_enabled and class_name:
                # The cascade governs the whole [cascade_floor, inf) range: it can
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
                    cascade_floor=cascade_floor,
                )
                if decision == Decision.AUTO_REJECT:
                    logger.debug(f"Cascade rejected '{class_name}' score={score:.3f}")
                    continue
            elif score < effective_threshold:
                # No cascade: plain per-class threshold filter.
                logger.debug(f"Dropping '{class_name}' score={score:.3f} < threshold={effective_threshold:.3f}")
                continue

            regions.extend(self._emit_regions(class_id, box_label, x1, y1, x2, y2, width, height, score))

        # RF-DETR runs NMS per class, not across classes, so the same facing
        # comes back once per plausible SKU. On the held-out clips every single
        # one of the SKU model's 31 false positives was such a duplicate:
        # collapsing them removed all 31 and cost no correct box and no correct
        # name. Do it before anything else consumes the boxes.
        self._dedupe_cross_class(image, regions)

        if shelf_tags_enabled and self.tag_class_map:
            self._apply_tag_corrections(image, regions)

        # Boxes the SKU model never drew, added unnamed, then named by template
        # match where the bank is confident. Measured on held-out clips these
        # two take the labeller from drawing 4.4 boxes an image to 0.9. Order
        # matters: proposals arrive after tag correction so a stale tag map
        # cannot name a box that nothing else has looked at yet.
        from control_models.box_proposals import BOX_PROPOSALS_ENABLED

        want_boxes = BOX_PROPOSALS_ENABLED if propose_boxes is None else propose_boxes
        if want_boxes:
            self._add_box_proposals(image, regions, width, height, name_proposals)

        # Naming runs last, over every box in the frame -- detections and
        # proposals alike. A proposal the template bank could not name is
        # exactly the box this can name, and it costs nothing extra: the call is
        # per frame, not per box.
        from cascade.box_naming import BOX_NAMING_ENABLED

        if BOX_NAMING_ENABLED if name_boxes is None else name_boxes:
            self._apply_box_naming(image, regions, width, height)

        return regions

    def _apply_box_naming(self, image, regions, width, height):
        """Replace the SKU on every box with what the shelf itself says it is.

        The class head's name is kept only where this stage declines AND the
        detector was confident (cascade.box_naming.HEAD_FLOOR); below that the
        box is left unnamed, which in this fork is a box the labeller has to
        name rather than one they have to notice is wrong. A failed call is not
        a declined one -- if the request never came back, the head's guesses
        stay exactly as they were.
        """
        from cascade import box_naming

        if not self.taxonomy_from_name:
            logger.debug("box naming needs a Taxonomy control; skipping")
            return
        rects = [r for r in regions if r.get("type") == "rectanglelabels"]
        if not rects:
            return
        boxes = [self._region_box_px(r, width, height) for r in rects]
        named = box_naming.name_boxes(image, boxes, self.class_names)
        if not named:
            return

        tax_by_region = {r["id"]: r for r in regions if r.get("type") == "taxonomy"}
        stale = []
        renamed = kept = cleared = 0
        for i, region in enumerate(rects):
            if i not in named:
                continue                      # the call failed for this chunk
            tax = tax_by_region.get(region["id"])
            path = self.taxonomy_path_map.get(named[i]) if named[i] else None
            if path:
                if tax is not None:
                    tax["value"]["taxonomy"] = [path]
                else:
                    regions.append({
                        "id": region["id"],
                        "from_name": self.taxonomy_from_name,
                        "to_name": self.taxonomy_to_name or self.to_name,
                        "type": "taxonomy",
                        "value": {"taxonomy": [path]},
                        "score": region.get("score", 0.0),
                    })
                renamed += 1
            elif tax is not None:
                if float(region.get("score") or 0.0) >= box_naming.HEAD_FLOOR:
                    kept += 1
                else:
                    stale.append(id(tax))
                    cleared += 1
        if stale:
            regions[:] = [r for r in regions if id(r) not in stale]
        logger.info(f"box naming: {renamed} named, {kept} left to the class head "
                    f"(>= {box_naming.HEAD_FLOOR}), {cleared} cleared")

    def _region_box_px(self, r, width, height):
        v = r["value"]
        x1 = v["x"] / 100.0 * width
        y1 = v["y"] / 100.0 * height
        return x1, y1, x1 + v["width"] / 100.0 * width, y1 + v["height"] / 100.0 * height

    def _dedupe_cross_class(self, image, regions, iou_thresh=0.6):
        """Collapse detections of the same facing under competing SKU names.

        Highest confidence wins, except when a cheap vision model is available
        and the competitors disagree -- then it is asked to pick, which is the
        one question it answers better than the detector's own ranking. On the
        held-out clips that arbitration turned 9 more names right than wrong.
        """
        rects = [r for r in regions if r.get("type") == "rectanglelabels"]
        if len(rects) < 2:
            return
        width, height = image.size
        tax_by_region = {r["id"]: r for r in regions if r.get("type") == "taxonomy"}

        kept, dropped = [], []
        for r in sorted(rects, key=lambda q: -q.get("score", 0.0)):
            box = self._region_box_px(r, width, height)
            clash = next((k for k in kept
                          if self._iou(box, self._region_box_px(k, width, height)) >= iou_thresh),
                         None)
            if clash is None:
                kept.append(r)
            else:
                dropped.append((clash, r))

        if not dropped:
            return

        if CASCADE_ENABLED:
            self._arbitrate(image, dropped, tax_by_region, width, height)

        # Each detection got its own region id in _emit_regions, and its
        # taxonomy row shares that id, so dropping by id removes the losing box
        # and its SKU together and cannot touch the winner.
        doomed_ids = {loser["id"] for _winner, loser in dropped}
        regions[:] = [r for r in regions if r.get("id") not in doomed_ids]
        logger.debug(f"cross-class dedupe removed {len(dropped)} duplicate box(es)")

    def _arbitrate(self, image, dropped, tax_by_region, width, height):
        """Let the vision model choose between competing names for one facing."""
        from cascade.gpt_tiebreaker import ask

        for winner, loser in dropped:
            names = []
            for r in (winner, loser):
                tax = tax_by_region.get(r["id"])
                if tax:
                    path = tax["value"].get("taxonomy", [[]])[0]
                    if path:
                        names.append(path[-1])
            if len(set(names)) < 2:
                continue
            box = self._region_box_px(winner, width, height)
            chosen = ask(image.crop(box), sorted(set(names)))
            win_tax = tax_by_region.get(winner["id"])
            if chosen and win_tax and chosen != names[0]:
                path = self.taxonomy_path_map.get(chosen)
                if path:
                    logger.debug(f"tiebreaker chose '{chosen}' over '{names[0]}'")
                    win_tax["value"]["taxonomy"] = [path]

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _add_box_proposals(self, image, regions, width, height, name_proposals=None):
        """Append facings the SKU model missed, named only where confident.

        A proposal with no SKU is the intended outcome, not a failure: the
        Taxonomy control is perRegion required, so an unnamed box cannot be
        submitted, and in this fork naming a proposed region is what accepts
        it. The labeller reads and types instead of reading, drawing and
        typing.
        """
        from control_models.box_proposals import propose
        from cascade.template_match import (TEMPLATE_MATCHING_ENABLED, name_crop)
        from control_models.base import MODEL_ROOT

        box_label = None
        available = list(self.control.labels_attrs.keys())
        if len(available) == 1:
            box_label = available[0]
        if box_label is None:
            logger.debug("box proposals need a single-label RectangleLabels; skipping")
            return

        taken = [self._region_box_px(r, width, height)
                 for r in regions if r.get("type") == "rectanglelabels"]
        for x1, y1, x2, y2, score in propose(image, taken):
            region_id = self.make_region_id()
            regions.append({
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
            })
            want_names = (TEMPLATE_MATCHING_ENABLED if name_proposals is None
                          else name_proposals)
            if not want_names or not self.taxonomy_from_name:
                continue
            name, _n = name_crop(image.crop((x1, y1, x2, y2)), MODEL_ROOT)
            path = self.taxonomy_path_map.get(name) if name else None
            if path:
                regions.append({
                    "id": region_id,
                    "from_name": self.taxonomy_from_name,
                    "to_name": self.taxonomy_to_name or self.to_name,
                    "type": "taxonomy",
                    "value": {"taxonomy": [path]},
                    "score": score,
                })

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
