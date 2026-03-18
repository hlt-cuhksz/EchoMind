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

echo "运行 LLaMA-Omni2 推理..."
python llama_omni2/inference/run_llama_omni2.py \
    --model_path models/"$model_name" \
    --question_file examples/questions.json \
    --answer_file "$output_dir"/answers.jsonl \
    --temperature 0 \

echo "运行 CoSy2 解码器生成音频..."
python llama_omni2/inference/run_cosy2_decoder.py \
    --input-path "$output_dir"/answers.jsonl \
    --output-dir "$output_dir"/wav \
    --lang en

echo "全部完成！输出目录：$output_dir"
