#!/bin/bash
# use uv to create venv with explicit python 3.10 for better wheel compatibility
uv venv --python 3.10 sdpo_env
source sdpo_env/bin/activate

# Install torch first so flash-attn can build
uv pip install "torch==2.4.0" torchvision torchaudio

# Install the package and normal dependencies
uv pip install -e .

# User requested uv, ruff, pytest
uv pip install ruff pytest

# Install key extra dependencies as tested for vLLM & LoRA
uv pip install "vllm>=0.8.5" "ray[default]>=2.41.0" peft 
uv pip install flash-attn --no-build-isolation
