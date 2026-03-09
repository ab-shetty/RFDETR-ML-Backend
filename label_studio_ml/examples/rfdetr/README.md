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
label_studio_ml/examples/rfdetr/models/
├── checkpoint_best_total.pth   ← model weights
└── checkpoint_best_total.txt   ← one class name per line
```

> These files are not in the repo (too large). Download them from shared storage or ask your team lead.

---

## 2. Configure environment

Copy the example env file and fill in your values:

```bash
cd label_studio_ml/examples/rfdetr
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
cd label_studio_ml/examples/rfdetr
docker-compose up --build
```

The backend will be available at `http://localhost:9091`.

**Connect to Label Studio**:
1. Open your project → Settings → Model
2. Add ML Backend: `http://localhost:9091`

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
cd label_studio_ml/examples/rfdetr
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
