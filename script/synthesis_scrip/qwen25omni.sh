conda activate qwen25omni



# ————————————————————Level 1. Understanding———————————————————————————————————————————————————————————
echo "Level 1: content understanding task (ASR)"
python src/eval-slm/eval_asr_main.py --root_dir echomind-master --data_type synthesis --model_name qwen25omni

echo "Level 1: voice understanding task (MCQ)"
python src/eval-slm/eval_mcq_main.py --root_dir echomind-master --data_type synthesis --model_name qwen25omni --MCQ_file understanding.json

# ————————————————————Level 2. Reasoning——————————————————————————————————————————————————————————————
echo "Level 2: integrated reasoning task (MCQ)"
python src/eval-slm/eval_mcq_main.py --root_dir echomind-master --data_type synthesis --model_name qwen25omni --MCQ_file reasoning.json

# ————————————————————Level 3. Conversation———————————————————————————————————————————————————————————
echo "Level 3: conversation (Open-domain Response)"

echo "system_prompt: qwen25omni_default"
python src/eval-slm/eval_response_main.py --root_dir echomind-master --data_type synthesis --model_name qwen25omni --system_prompt qwen25omni_default --voice_type target neutral alternative --audio_output
echo "system_prompt: basic"
python src/eval-slm/eval_response_main.py --root_dir echomind-master --data_type synthesis --model_name qwen25omni --system_prompt basic --voice_type target neutral alternative --audio_output
echo "system_prompt: enhance"
python src/eval-slm/eval_response_main.py --root_dir echomind-master --data_type synthesis --model_name qwen25omni --system_prompt enhance --voice_type target neutral alternative --audio_output
