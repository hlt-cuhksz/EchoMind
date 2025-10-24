conda activate step-audio


# ————————————————————Level 1. Understanding———————————————————————————————————————————————————————————
echo "Level 1: content understanding task (ASR)"
python src/eval-slm/eval_asr_main.py --root_dir echomind-master --data_type human --model_name step-audio-chat --api_key ""

echo "Level 1: voice understanding task (MCQ)"
python src/eval-slm/eval_mcq_main.py --root_dir echomind-master --data_type human --model_name step-audio-chat --api_key "" --MCQ_file understanding.json

# ————————————————————Level 2. Reasoning——————————————————————————————————————————————————————————————
echo "Level 2: integrated reasoning task (MCQ)"
python src/eval-slm/eval_mcq_main.py --root_dir echomind-master --data_type human --model_name step-audio-chat --api_key "" --MCQ_file reasoning.json

# ————————————————————Level 3. Conversation———————————————————————————————————————————————————————————
echo "Level 3: conversation (Open-domain Response)"

echo "system_prompt: no"
python src/eval-slm/eval_response_main.py --root_dir echomind-master --data_type human --model_name step-audio-chat --api_key "" --system_prompt no --voice_type target neutral alternative --audio_output
echo "system_prompt: basic"
python src/eval-slm/eval_response_main.py --root_dir echomind-master --data_type human --model_name step-audio-chat --api_key "" --system_prompt basic --voice_type target neutral alternative --audio_output
echo "system_prompt: enhance"
python src/eval-slm/eval_response_main.py --root_dir echomind-master --data_type human --model_name step-audio-chat --api_key "" --system_prompt enhance --voice_type target neutral alternative --audio_output

