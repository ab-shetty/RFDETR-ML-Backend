import os
import logging

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from control_models.base import ControlModel
from control_models.rectangle_labels import RFDETRRectangleLabelsModel
from typing import List, Dict, Optional


logger = logging.getLogger(__name__)
if not os.getenv("LOG_LEVEL"):
    logger.setLevel(logging.INFO)

available_model_classes = [
    RFDETRRectangleLabelsModel,
]


class RFDETR(LabelStudioMLBase):
    """Label Studio ML Backend based on RF-DETR"""

    def setup(self):
        self.set("model_version", self._version())

    @staticmethod
    def _version() -> str:
        """What produced a prediction: the checkpoint AND the pipeline around it.

        This was the constant string "rfdetr" through every retrain and every
        change to the cascade, so Label Studio could not tell a prediction made
        by the July 179-class model from one made today -- and anything that
        skips re-prediction when the version matches would serve the old one
        forever. Stale predictions are invisible precisely because the field
        that exists to expose them never moved.

        The flags belong in it as much as the weights do: the same checkpoint
        with box proposals on produces materially different regions.
        """
        import json
        import os

        from control_models.base import CASCADE_ENABLED, MODEL_ROOT, SHELF_TAGS_ENABLED

        run = "unknown-run"
        deployed = os.path.join(MODEL_ROOT, "deployed.json")
        if os.path.exists(deployed):
            try:
                run = json.load(open(deployed)).get("run_name") or run
            except Exception:
                pass
        else:
            ckpt = os.path.join(MODEL_ROOT, "checkpoint_best_total.pth")
            if os.path.exists(ckpt):
                run = f"unrecorded-{int(os.path.getmtime(ckpt))}"

        from cascade.template_match import TEMPLATE_MATCHING_ENABLED
        from control_models.box_proposals import BOX_PROPOSALS_ENABLED

        flags = "".join(f"+{n}" for n, on in (
            ("cascade", CASCADE_ENABLED),
            ("tags", SHELF_TAGS_ENABLED),
            ("box", BOX_PROPOSALS_ENABLED),
            ("tpl", TEMPLATE_MATCHING_ENABLED),
        ) if on)
        return f"{run}{flags}"

    def detect_control_models(self) -> List[ControlModel]:
        control_models = []

        for control in self.label_interface.controls:
            if not control.to_name:
                logger.warning(f'{control.tag} {control.name} has no "toName" attribute, skipping')
                continue

            for model_class in available_model_classes:
                if model_class.is_control_matched(control):
                    instance = model_class.create(self, control)
                    if not instance:
                        continue
                    available_labels = list(control.labels_attrs.keys())
                    if not instance.label_map and len(available_labels) != 1:
                        logger.error(
                            f"No label map built for '{control.tag}' control tag '{instance.from_name}'.\n"
                            f"Ensure your Label Studio labels match the model class names,\n"
                            f"or use a single label to map all detections to it (e.g. 'Product').\n"
                            f"Labels in your config: {available_labels}\n"
                            f"Model class names: {instance.class_names}"
                        )
                        continue
                    control_models.append(instance)
                    logger.debug(f"Control tag with model detected: {instance.type} -> {instance.from_name}")
                    break

        if not control_models:
            raise ValueError(
                f"No suitable RectangleLabels control tags connected to Image tags "
                f"detected in label config:\n{self.label_config}"
            )

        return control_models

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        # cascade_mode ("off" | "cascade" | "cascade_shelf_tags"), when present,
        # overrides the CASCADE_ENABLED/SHELF_TAGS_ENABLED env vars for this
        # call only -- see Label Studio's "Retrieve Predictions" action
        # (data_manager/actions/basic.py), which is the only caller that sets
        # it. Absent (None), predict_regions falls back to the env-var
        # defaults this process started with.
        # detection_floor likewise overrides the detection cutoff for this call
        # only, so a user can trade recall against precision from the UI without
        # restarting the backend. It is a single knob covering both the cascade
        # floor and the plain-mode threshold -- see predict_regions for why.
        cascade_mode = (context or {}).get("cascade_mode")
        detection_floor = (context or {}).get("detection_floor")
        # Box proposals and template naming are separate axes from the cascade:
        # a run can want boxes without guessed names (a new store visit, where
        # most SKUs have no template and a name would just be wrong) or the
        # bare detector for comparison. None means "use the env default".
        propose_boxes = (context or {}).get("propose_boxes")
        name_proposals = (context or {}).get("name_proposals")
        logger.info(
            f"Run prediction on {len(tasks)} tasks, project ID = {self.project_id}, "
            f"cascade_mode={cascade_mode}, detection_floor={detection_floor}"
        )
        control_models = self.detect_control_models()

        predictions = []
        for task in tasks:
            regions = []
            for model in control_models:
                path = model.get_path(task)
                regions += model.predict_regions(
                    path, cascade_mode=cascade_mode, detection_floor=detection_floor,
                    propose_boxes=propose_boxes, name_proposals=name_proposals
                )

            all_scores = [r["score"] for r in regions if "score" in r]
            avg_score = sum(all_scores) / max(len(all_scores), 1)

            predictions.append({
                "result": regions,
                "score": avg_score,
                "model_version": self.model_version,
            })

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """Training is handled by train_local.py (runs natively with MPS on Apple Silicon)."""
        logger.info(
            f"Received event '{event}'. To fine-tune, run: python train_local.py"
        )
        return {}
