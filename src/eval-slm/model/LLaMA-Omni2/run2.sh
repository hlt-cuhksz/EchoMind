#!/bin/bash

# 设置模型名称
model_name=LLaMA-Omni2-7B

# 检查是否提供了模型名称
if [ -z "$model_name" ]; then
  echo "用法: ./run.sh <model_name>"
  exit 1
fi

# 创建输出目录
output_dir=examples/$model_name
mkdir -p "$output_dir"

python llama_omni2.py \
    --audio /share/workspace/EQ-SLM/EQ-Bench/src/03-eval-slm/models/LLaMA-Omni2/examples/wav/helpful_base_7.wav \
    --output_path output.wav \
