#!/bin/bash
#SBATCH -J deploy-speech-model  # Job name
#SBATCH -c 8                    # CPU cores
#SBATCH -p gpu3node             # GPU partition
#SBATCH --gres=gpu:1            # Request 1 GPU
#SBATCH --mem=32G               # Memory allocation
#SBATCH -t 0-00:05:00           # Maximum runtime

module load cuda/11.8           # Load CUDA 11.8 module
module load anaconda3           # Load anaconda 3 module

eval "$(conda shell.bash hook)"
conda activate eq-bench-audio-flamingo-3

python llava/cli/infer_audio.py --model-base /share/workspace/EQ-SLM/EQ-Bench/models/audio-flamingo-3 --conv-mode auto --text "Please describe the audio in detail" --media static/audio/audio2.wav
