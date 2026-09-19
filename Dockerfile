FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN set -eux \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY src /opt/flaird/src
COPY pyproject.toml /opt/flaird
COPY README.md LICENSE /opt/flaird/
WORKDIR /opt/flaird
RUN set -eux \
    && python3 -m pip config set global.break-system-packages true \
    && python3 -m pip install --no-cache . \
    && rm -rf ./build ./*.egg-info

ENV HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/flaird"]
