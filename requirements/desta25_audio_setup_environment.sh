#!/bin/bash

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

    conda create -n $CONDA_ENV python=3.10.18 -y
    conda activate $CONDA_ENV
else
    echo "Skipping conda environment creation. Make sure you have the correct environment activated."
fi

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip3 install torch torchaudio torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu118
pip install whisper_normalizer huggingface_hub transformers==4.49.0 peft
pip install librosa
pip install -e . 

pip install "numpy<2.0" 
pip install backoff 
