# RF-DETR Label Studio ML Backend

Object detection ML backend for [Label Studio](https://labelstud.io/) using [RF-DETR](https://github.com/roboflow/rf-detr).

- **Inference**: runs in Docker
- **Fine-tuning**: runs natively on macOS with Apple Silicon (MPS GPU)

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (for training only)
- A running [Label Studio](https://labelstud.io/) instance
- The RF-DETR model checkpoint (`.pth`) and class names file (`.txt`) — get these from your team

---

## 1. Get the model files

Place the following files in the `models/` directory (create it if it doesn't exist):

```
label_studio_ml/examples/models/
├── checkpoint_best_total.pth   ← model weights
└── checkpoint_best_total.txt   ← one class name per line
```

> These files are not in the repo (too large). Download them from shared storage or ask your team lead.

---

## 2. Configure environment

Copy the example env file and fill in your values:

```bash
cd label_studio_ml/examples
cp .env.example .env
```

Edit `.env`:

```
LABEL_STUDIO_API_KEY=<your Label Studio API key>
PROJECT_ID=<your Label Studio project number>
```

**Getting your API key**: Label Studio → click your avatar → Account & Settings → Access Token

**Getting your Project ID**: open your project in the browser — the number in the URL (`/projects/7/`) is the ID.

> Do NOT set `LABEL_STUDIO_HOST` in `.env` — Docker uses `host.docker.internal:8080` by default, and the training script uses `localhost:8080`.

---

## 3. Run inference (Docker)

```bash
cd label_studio_ml/examples
docker-compose up --build
```

The backend will be available at `http://localhost:9091`.

**Connect to Label Studio**:
1. Open your project → Settings → Model
2. Add ML Backend: `http://localhost:9091`

---

## Per-class confidence thresholds

`MODEL_SCORE_THRESHOLD` (default `0.5`) is a single global cutoff applied to every class. In practice, per-class precision/recall varies a lot — some classes are reliable at 0.5, others produce mostly false positives even there. To calibrate per class instead:

```bash
python scripts/compute_class_thresholds.py \
    --checkpoint models/checkpoint_best_total.pth \
    --valid-dir /path/to/coco/valid/split
```

This runs real inference over the validation split, sweeps a threshold grid per class, and writes `models/class_thresholds.json` (threshold + precision/recall/F-beta per class). The backend picks it up automatically on next load — no code change needed. Classes with too few validation instances (default: fewer than 3) fall back to `MODEL_SCORE_THRESHOLD` rather than trust a noisy estimate; check the script's output for how many classes actually got tuned vs. defaulted. Re-run after every training run as validation data grows.

An explicit `model_score_threshold` set on the `<RectangleLabels>` labeling-config tag still works — it now acts as a floor on top of the per-class value (never goes below what's explicitly configured), rather than replacing per-class tuning entirely.

---

## Box proposals + template naming

The SKU detector only detects what it recognises, so a facing it never boxes is
one it already declined to identify — asked to classify such a crop it gets 7 of
79 right, against 77% on boxes it found itself. Two stages cover that gap.

1. **Box proposals** (`control_models/box_proposals.py`, `BOX_PROPOSALS_ENABLED`)
   — a class-agnostic facing detector proposes rectangles the SKU model did not
   draw, deduplicated against the ones it did. Proposals carry **no SKU on
   purpose**: the Taxonomy control is `perRegion required`, so an unnamed box
   cannot be submitted, and in this fork naming a proposed region is what
   accepts it. The labeller reads and types instead of reading, drawing and
   typing.
2. **Template naming** (`cascade/template_match.py`, `TEMPLATE_MATCHING_ENABLED`)
   — ORB keypoint matching against per-instance labelled crops
   (`models/template_bank.npz`, built by `scripts/build_template_bank.py`) names
   those proposals where it is confident. `TEMPLATE_MIN_MATCHES` defaults to 18
   because the stage is **never right about a SKU absent from its bank** — there
   is nothing to match — so the bar exists to keep it silent there rather than
   to trade accuracy.

Both disable themselves with a warning when their model file is missing, so a
machine without them degrades to plain detection rather than erroring. Also
always on: cross-class dedupe, since RF-DETR runs NMS per class but not across
classes — every one of the SKU model's 31 false positives on the held-out clips
was the same facing under a competing name.

Measured with `tj-labeling-ops/pipeline_dryrun.py`, which scores the whole
pipeline offline as labelling work (boxes to draw, strays to delete, names to
fix) rather than as mAP. Read its caveats before quoting any number from it:
most of the current eval data overlaps the training set.

## Naming boxes from the shelf (`cascade/box_naming.py`, `BOX_NAMING_ENABLED`)

The class head learns SKU names from labelled shelves, so it names a store it has
never seen about as well as its training data allows: on the two held-out visits
it got **51%** of the boxes it found right, and most of that came from a narrow
high-confidence band. The shelf tag under each product names it independently of
any training data — the store reprints it whenever the shelf changes.

An earlier attempt used tags by reading them in isolation and working out which
box each named from its position. That pairing became the dominant error (dense
shelves, approximate coordinates, most wrong names landing one slot over), and
capped the whole path at 30–45%. Drawing the numbered boxes on the frame removes
the pairing step: the question stops being "where is this tag" and becomes "what
is in box 7", which the model answers from the tag *and* the packaging.

Measured against human labels on both held-out store visits
(`scripts/eval_box_naming.py`):

| | class head | tags paired by geometry | boxes drawn on the frame |
|---|---|---|---|
| Coppell (sharp footage) | 51% | 45% | **87%** |
| Laguna (soft footage)   | 51% | 19% | **73%** |

`scripts/eval_naming_gate.py` answers whether the class head is still worth
consulting: it only catches up above ~0.6 confidence, which is a quarter of its
boxes, and of 183 disagreements the vision model was right in 160. So the head's
name is kept **only** where this stage declines *and* the detector was confident
(`BOX_NAMING_HEAD_FLOOR`, default 0.8). Below that the box is left unnamed —
work the labeller does, rather than a wrong name they have to notice.

Runs last, over detections and proposals alike, so a proposal the template bank
could not name is named here at no extra cost (one call per frame, not per box,
~$0.01). `gpt-5.6-terra` rather than the cheaper Luna because the tags are often
handwritten, which is a capability-tier failure for nano-tier models — the same
finding `scripts/build_tag_index.py` documents for the reader. A failed call is
not a declined one: if the request never comes back, the head's guesses stay.

Both harnesses score naming on ground-truth boxes, so they measure naming in
isolation; `--boxes detector` runs the production condition, where a missed
facing is never named and a stray one is named for nothing.

**`tag_class_map.json` is gone, along with the stage that used it.** It resolved
half the tags on Laguna and got three quarters of those wrong (`'APPLE JUICE' ->
'Matcha Green Tea'`): built by pairing tags to already-labelled boxes on one
visit, so that pairing's noise became majority votes, and structurally unable to
name a SKU nobody had labelled yet. Removed rather than left switchable —
`SHELF_TAGS_ENABLED`, `_apply_tag_corrections`, `lookup_class`,
`propose_from_tags`, `build_tag_class_map.py` and the map itself. What survives
is `cascade/shelf_tags.detect_tags`, the reader, which is still the cheapest way
to find out what is on a shelf; `scripts/eval_tag_naming.py` is the record of why
the rest went.

## Verification cascade (OCR + embedding match + gpt-5.6-luna)

Per-class thresholds alone can't fix a class the model barely has data for — no threshold rescues a detector that's never learned to recognize something reliably. The cascade adds two cheap, fast signals plus a gpt-5.6-luna tiebreaker used only when they disagree, filtering junk detections *before* they become Label Studio pre-annotations rather than after a human deletes them:

1. **OCR** (`cascade/ocr.py`) — reads text off the crop, fuzzy-matches against the class's expected label text. Only meaningful for branded/text-bearing products; produce and generic dairy classes have no expected text, so OCR contributes no signal there by design.
2. **Embedding match** (`cascade/embedding_match.py`) — embeds the crop with the RF-DETR backbone and compares against a per-class reference gallery built from already-labeled crops.
3. **gpt-5.6-luna** (`cascade/gpt_tiebreaker.py`) — called *only* when OCR and embedding-match disagree with each other. Not run on every detection — that would be slow and expensive at real label-batch volume.

Decision policy (`cascade/pipeline.py`): below the per-class threshold → reject outright; no signal available for this class at all → trust the threshold; every available signal agrees → accept; every signal disagrees → reject; signals disagree with *each other* → ask gpt-5.6-luna.

**Setup:**

```bash
python scripts/build_ocr_expected_text.py --master-list /path/to/master_list.csv
python scripts/build_reference_gallery.py --checkpoint models/checkpoint_best_total.pth \
    --dataset-dir /path/to/training-data/rf-detr-combined
```

Normally you do not run the gallery step by hand: `tj-labeling-ops/deploy_checkpoint.py`
rebuilds it against the new weights whenever a checkpoint is adopted, because a
gallery built from a *different* checkpoint's backbone is not comparable to the
one being served.

Then set `CASCADE_ENABLED=true` and `OPENAI_API_KEY=<key>` in `.env`. Off by default — it adds real per-detection latency (OCR + a backbone forward pass, occasionally a gpt-5.6-luna call), so turn it on once the resource files above actually exist; before that it's a silent no-op.

---

## 4. Fine-tune the model (Apple Silicon only)

### Setup (one-time)

```bash
conda create -n rfdetr python=3.10 -y
conda activate rfdetr
pip install rfdetr label-studio-sdk python-dotenv requests Pillow torch torchvision
```

### Run training

```bash
cd label_studio_ml/examples
conda activate rfdetr
PYTORCH_ENABLE_MPS_FALLBACK=1 python train_local.py
```

This will:
1. Fetch all annotated tasks from your Label Studio project
2. Build a COCO-format dataset from the annotations
3. Fine-tune the RF-DETR model using your Mac's GPU (MPS)
4. Save the new checkpoint to `models/checkpoint_best_total.pth`

### Load the new model

After training completes, restart Docker to load the updated checkpoint:

```bash
docker-compose down && docker-compose up
```

### Training options

Set these in `.env` or as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAINING_EPOCHS` | `10` | Number of fine-tuning epochs |
| `MODEL_FILE` | `checkpoint_best_total.pth` | Checkpoint filename in `models/` |
| `MODEL_ROOT` | `./models` | Path to models directory |

---

## Architecture

```
Label Studio ──→ Docker (port 9091) ──→ RF-DETR inference
                                              ↑
                                        models/checkpoint_best_total.pth
                                              ↑
                              train_local.py (native MPS training)
```

The `models/` directory is mounted as a Docker volume, so training output is immediately available to the container after restart.
