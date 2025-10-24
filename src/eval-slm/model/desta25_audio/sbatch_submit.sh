#!/bin/bash
#SBATCH -J speech-model         # Job name
#SBATCH -c 8                    # CPU cores
#SBATCH -p gpu3node             # GPU partition
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --mem=32G               # Memory allocation
#SBATCH -t 0-02:00:00           # Maximum runtime

module load cuda/11.8                           # Load CUDA 11.8 module
module load anaconda3                           # Load anaconda 3 module

eval "$(conda shell.bash hook)"
conda activate eq-bench-desta-2.5-audio

# ip_address="10.20.12.42"
# export http_proxy=http://$ip_address:10808
# export https_proxy=http://$ip_address:10808
# export all_proxy=socks5://$ip_address:10808
# export HTTP_PROXY=http://$ip_address:10808
# export HTTPS_PROXY=http://$ip_address:10808
# export ALL_PROXY=socks5://$ip_address:10808

python infer.py
