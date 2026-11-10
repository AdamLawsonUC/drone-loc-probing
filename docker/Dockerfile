# Flight server and activation-extraction image for the university A100.
#
# The base image carries torch with CUDA 12.1, chosen for wolfgang (driver 530, CUDA 12.1 at most).
# If the host driver is ever upgraded to 550+, pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime also works
# (the laptop install picks its own CUDA tag from the README).
#
# Build:  docker build -t drone-loc-probing -f docker/Dockerfile .
# Run:    docker/run_server.sh --dry-run            (see that script for the mounts)

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /work
EXPOSE 8765

# weights and code are mounted, not copied: -v $HOME/.cache/huggingface:/hf  -v $(pwd):/work
CMD ["python", "scripts/80_flight_server.py", "--dry-run", "--port", "8765"]