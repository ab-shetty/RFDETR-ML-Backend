FROM python:3.10-slim
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    apt-get -y update \
    && apt-get install -y git wget curl gcc \
    && apt-get install -y libsm6 libxext6 libffi-dev python3-dev libgl1 libglib2.0-0 \
    && apt-get install -y tesseract-ocr

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9091 \
    WORKERS=1 \
    THREADS=4

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install model requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

WORKDIR /app

COPY . ./

ENV PYTHONPATH=/app

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
