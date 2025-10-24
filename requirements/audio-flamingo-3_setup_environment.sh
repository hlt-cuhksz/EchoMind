#!/bin/bash
# Modified from https://github.com/NVIDIA/audio-flamingo/blob/audio_flamingo_3/environment_setup.sh

set -e

echo "========================================================"
echo "                      NOTICE"
echo "========================================================"
echo "1. This script is designed to be run from the root of"
echo "   the model's code repository."
echo "2. The model is evaluated with a specific CUDA version. "
echo "   If you use a different version, please update the PyTorch URL."
echo "3. Provide a positional argument as the conda environment"
echo "   name. Otherwise, the script will modify your current "
echo "   active conda environment."
echo "========================================================"
echo ""

read -p "Have you read and understood the notice above? (y/n): " confirmation

if [[ "$confirmation" != "y" ]]; then
  echo "Confirmation not provided. Exiting script."
  exit 1
fi

echo ""
echo "Confirmation received. Proceeding with the script..."

# PYPI_MIRROR_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"

CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    # This is required to activate conda environment
    eval "$(conda shell.bash hook)"

    conda create -n $CONDA_ENV python=3.10.14 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

echo "[INFO] Using python $(which python)"
echo "[INFO] Using pip $(which pip)"

# This is required to enable PEP 660 support
pip install --upgrade pip setuptools 
pip install -e ".[train,eval]" 

pip install hydra-core loguru Pillow pydub 

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pip3 install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -f https://mirrors.aliyun.com/pytorch-wheels/cu128

# Install FlashAttention2
pip install flash_attn==2.7.3 
pip install transformers==4.46.0 
pip install pytorchvideo==0.1.5 
pip install deepspeed==0.15.4 
pip install accelerate==0.34.2 
pip install numpy==1.26.4 
pip install opencv-python-headless==4.8.0.76 
pip install matplotlib 
# numpy introduce a lot dependencies issues, separate from pyproject.yaml


# audio
pip install soundfile librosa openai-whisper ftfy 
pip install ffmpeg 
pip install jiwer 
pip install wandb 
pip install kaldiio 
pip install peft==0.14.0 
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')

# Downgrade protobuf to 3.20 for backward compatibility
pip install protobuf==3.20.* 

# Replace transformers and deepspeed files
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

pip install triton==3.1.0 

pip install backoff 
