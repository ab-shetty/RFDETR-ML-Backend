#!/usr/bin/env bash
# One-time setup: install the RF-DETR ML backend's system + Python deps into
# a local venv, so it can run as a plain background process instead of a
# Docker container.
#
# Why native instead of Docker for this piece specifically (not for Label
# Studio -- see label-studio-fork's GHCR CI workflow): this backend has no
# heavy build step of its own (no frontend to compile), so a container's
# main benefit -- doing an expensive build once and distributing the
# artifact -- doesn't apply here. What's left is Docker's overhead: full
# base-OS layers and duplicated Python packages, several GB for something a
# venv does just as reproducibly with a documented, idempotent install
# script (this one). It also removes a footgun: when Label Studio runs
# natively too (see tj-labeling-ops), a Docker-only backend forces the
# host.docker.internal vs localhost distinction on every session; native on
# both sides means everything is just localhost.
#
# Usage: ./scripts/setup_native_venv.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLES_DIR="$REPO_ROOT/label_studio_ml/examples"
VENV_DIR="$REPO_ROOT/.venv"

echo "==> Installing system packages (apt, needs sudo)"
sudo apt-get update -qq
sudo apt-get install -y -q \
  git wget curl gcc \
  libsm6 libxext6 libffi-dev python3-dev libgl1 libglib2.0-0 \
  tesseract-ocr

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "==> Installing base requirements"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$EXAMPLES_DIR/requirements-base.txt"

echo "==> Installing CPU-only torch (the default PyPI wheel is a ~2.7GB CUDA build unused on CPU-only hosts)"
"$VENV_DIR/bin/pip" install -r "$EXAMPLES_DIR/requirements-torch-cpu.txt" --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing model requirements"
"$VENV_DIR/bin/pip" install -r "$EXAMPLES_DIR/requirements.txt"

echo
echo "Done. Venv at $VENV_DIR"
echo "tj-labeling-ops/session_start.sh runs the backend from this venv automatically."
