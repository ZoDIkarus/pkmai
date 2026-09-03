FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        python3-dev \
        libgl1 \
        libglib2.0-0 \
        libsdl2-2.0-0 \
        libsdl2-dev \
        libpng-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && grep -v '^torch$' requirements.txt > /tmp/requirements-without-torch.txt \
    && python -m pip install -r /tmp/requirements-without-torch.txt

COPY src ./src
COPY scripts ./scripts

ENTRYPOINT ["python", "-u", "src/train.py"]
