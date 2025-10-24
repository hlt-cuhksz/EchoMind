#!/bin/bash

set -e
set -x

timestamp="$1"
if [ -z "$timestamp" ]
then
	timestamp=`date +'%Y%m%d_%H%M%S'`
fi

######################################################################
export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/VITA-Audio/

export LOCAL_ROOT_PATH=/data_local/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/VITA-Audio/
mkdir -p ${LOCAL_ROOT_PATH}
mkdir -p ${LOCAL_CODE_PATH}

apt install -y rsync
mkdir -p ${LOCAL_CODE_PATH}
rsync -a --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/

cd ${LOCAL_CODE_PATH}
rm -fr datasets
ln -s ${ROOT_PATH}/data datasets

######################################################################
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${CODE_PATH}/scripts/set_env_ds_gpu.sh
pip3 install transformers==4.48.3
#pip3 install --no-index --find-links=/data/software/ transformers==4.48.3

######################################################################
OUTPUT_DIR=${ROOT_PATH}/output/LM/"$0"/${timestamp}/

mkdir -p ${OUTPUT_DIR}
rsync -avh $0 ${OUTPUT_DIR}

export HF_HOME="${ROOT_PATH}/data/HF_HOME/"
mkdir -p ${HF_HOME}
export HF_ENDPOINT=https://hf-mirror.com

export MODELSCOPE_CACHE="${ROOT_PATH}/data/MODELSCOPE_CACHE/"
mkdir -p ${MODELSCOPE_CACHE}

export LC_ALL="en_US.utf8"

######################################################################
LOG=${OUTPUT_DIR}/log_node${INDEX}.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


######################################################################
if true
#if false
then
	MODEL_NAME_OR_PATH="/data/output/LM/scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage2.sh/VITA-Audio-Boost/"
	MODEL_NAME_OR_PATH="/data/output/LM/scripts/deepspeed/sts_qwen25/finetune_glm4voice_mtp10_stage2.sh/VITA-Audio-Balance/"

	AUDIO_TOKENIZER_PATH=${ROOT_PATH}/models/THUDM/glm-4-voice-tokenizer
	FLOW_PATH=${ROOT_PATH}/models/THUDM/glm-4-voice-decoder
	AUDIO_TOKENIZER_TYPE="glm4voice"

	export PYTHONPATH=${PYTHONPATH}:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice/:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice/cosyvoice/:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice/third_party/Matcha-TTS/

fi

######################################################################
DISTRIBUTED_ARGS="
--nproc_per_node $NPROC_PER_NODE \
	--nnodes $NNODES \
	--node_rank $NODE_RANK \
	--master_addr $MASTER_ADDR \
	--master_port $MASTER_PORT
	"

######################################################################
if true
#if false
then
	apt-get update && apt install -y ffmpeg

	JSON_PATH=${ROOT_PATH}/data/jsonl/fixie-ai/llama-questions/test.jsonl
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_sqa.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/llama-questions/

	python evaluation/compute-acc-of-contain.py ${OUTPUT_DIR}/llama-questions/test_hyp_ref_text.json
	echo "copypaste ACC: ${JSON_PATH}"
	python evaluation/compute-acc-of-contain.py ${OUTPUT_DIR}/llama-questions/test_hyp_ref_speech.json
	echo "copypaste ACC: ${JSON_PATH}"


	JSON_PATH=${ROOT_PATH}/data/jsonl/fixie-ai/trivia_qa-audio/validation.jsonl
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_sqa.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/trivia_qa-audio/

	python evaluation/compute-acc-of-contain.py ${OUTPUT_DIR}/trivia_qa-audio/validation_hyp_ref_text.json
	echo "copypaste ACC: ${JSON_PATH}"
	python evaluation/compute-acc-of-contain.py ${OUTPUT_DIR}/trivia_qa-audio/validation_hyp_ref_speech.json
	echo "copypaste ACC: ${JSON_PATH}"


	JSON_PATH=${ROOT_PATH}/data/jsonl/fixie-ai/spoken-web-questions/test.jsonl
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_sqa.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/spoken-web-questions/

	python evaluation/compute-acc-of-contain.py ${OUTPUT_DIR}/spoken-web-questions/test_hyp_ref_text.json
	echo "copypaste ACC: ${JSON_PATH}"
	python evaluation/compute-acc-of-contain.py ${OUTPUT_DIR}/spoken-web-questions/test_hyp_ref_speech.json
	echo "copypaste ACC: ${JSON_PATH}"

fi


######################################################################
if true
#if false
then
	JSON_PATH=${ROOT_PATH}/data/jsonl/fixie-ai/librispeech_asr/validation.clean.jsonl

	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_asr.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/librispeech_asr/

	#python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/validation.clean_ref.txt ${OUTPUT_DIR}/librispeech_asr/validation.clean_hyp.txt
	#echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/validation.clean_ref.txt ${OUTPUT_DIR}/librispeech_asr/validation.clean_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"

	JSON_PATH=${ROOT_PATH}/data/jsonl/fixie-ai/librispeech_asr/validation.other.jsonl

	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_asr.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/librispeech_asr/

	#python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/validation.other_ref.txt ${OUTPUT_DIR}/librispeech_asr/validation.other_hyp.txt
	#echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/validation.other_ref.txt ${OUTPUT_DIR}/librispeech_asr/validation.other_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"


	JSON_PATH=${ROOT_PATH}/data/jsonl/fixie-ai/librispeech_asr/test.clean.jsonl

	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_asr.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/librispeech_asr/

	#python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/test.clean_ref.txt ${OUTPUT_DIR}/librispeech_asr/test.clean_hyp.txt
	#echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/test.clean_ref.txt ${OUTPUT_DIR}/librispeech_asr/test.clean_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"

	JSON_PATH=${ROOT_PATH}/data/jsonl/fixie-ai/librispeech_asr/test.other.jsonl

	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_asr.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/librispeech_asr/

	#python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/test.other_ref.txt ${OUTPUT_DIR}/librispeech_asr/test.other_hyp.txt
	#echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/librispeech_asr/test.other_ref.txt ${OUTPUT_DIR}/librispeech_asr/test.other_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"

fi


######################################################################
if true
#if false
then
	JSON_PATH=${ROOT_PATH}/data/jsonl/wenet-e2e/wenetspeech/TEST_MEETING.jsonl
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_asr.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/wenetspeech/

	python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/wenetspeech/TEST_MEETING_ref.txt ${OUTPUT_DIR}/wenetspeech/TEST_MEETING_hyp.txt
	echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/wenetspeech/TEST_MEETING_ref.txt ${OUTPUT_DIR}/wenetspeech/TEST_MEETING_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"

	JSON_PATH=${ROOT_PATH}/data/jsonl/wenet-e2e/wenetspeech/TEST_NET.jsonl
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_asr.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/wenetspeech/

	python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/wenetspeech/TEST_NET_ref.txt ${OUTPUT_DIR}/wenetspeech/TEST_NET_hyp.txt
	echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/wenetspeech/TEST_NET_ref.txt ${OUTPUT_DIR}/wenetspeech/TEST_NET_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"
fi


######################################################################
if true
#if false
then
	JSON_PATH=${ROOT_PATH}/data/jsonl/shenyunhang/AISHELL-1/test.jsonl

	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_asr.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/AISHELL-1/

	#python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/AISHELL-1/_test.clean_ref.txt ${OUTPUT_DIR}/AISHELL-1/test.clean_hyp.txt
	#echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/AISHELL-1/test_ref.txt ${OUTPUT_DIR}/AISHELL-1/test_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"


fi


######################################################################
if true
#if false
then
	JSON_PATH=${ROOT_PATH}/data/jsonl/mythicinfinity/libritts/test.clean.jsonl
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_libritts.py \
		--json_path ${JSON_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/libritts/ \

	#python evaluation/compute-cer.py --char=1 --v=1 ${OUTPUT_DIR}/libritts/test.clean_ref.txt ${OUTPUT_DIR}/libritts/test.clean_hyp.txt
	#echo "copypaste CER: ${JSON_PATH}"
	python evaluation/compute-wer.py --char=1 --v=1 ${OUTPUT_DIR}/libritts/test.clean_ref.txt ${OUTPUT_DIR}/libritts/test.clean_hyp.txt
	echo "copypaste WER: ${JSON_PATH}"
fi


######################################################################
if true
#if false
then

	DATA_PATH=${ROOT_PATH}/data/BytedanceSpeech/seed-tts-eval/
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_seedtts.py \
		--data_path ${DATA_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/seed-tts/ \
		--speaker_prompt \

	export ARNOLD_WORKER_GPU=${NPROC_PER_NODE}
	cd ${LOCAL_CODE_PATH}/third_party/seed-tts-eval

	bash cal_wer.sh ${DATA_PATH}/seedtts_testset/zh/meta.lst ${OUTPUT_DIR}/seed-tts/zh/ zh
	echo "copypaste WER: ${DATA_PATH} zh"
	bash cal_wer.sh ${DATA_PATH}/seedtts_testset/zh/hardcase.lst ${OUTPUT_DIR}/seed-tts/hardcase/ zh
	echo "copypaste WER: ${DATA_PATH} hardcase"
	bash cal_wer.sh ${DATA_PATH}/seedtts_testset/en/meta.lst ${OUTPUT_DIR}/seed-tts/en/ en
	echo "copypaste WER: ${DATA_PATH} en"

	bash cal_sim.sh ${DATA_PATH}/seedtts_testset/zh/meta.lst ${OUTPUT_DIR}/seed-tts/zh/ ${DATA_PATH}/wavlm_large_finetune.pth
	echo "copypaste SIM: ${DATA_PATH} zh"
	bash cal_sim.sh ${DATA_PATH}/seedtts_testset/zh/hardcase.lst ${OUTPUT_DIR}/seed-tts/hardcase/ ${DATA_PATH}/wavlm_large_finetune.pth
	echo "copypaste SIM: ${DATA_PATH} hardcase"
	bash cal_sim.sh ${DATA_PATH}/seedtts_testset/en/meta.lst ${OUTPUT_DIR}/seed-tts/en/ ${DATA_PATH}/wavlm_large_finetune.pth
	echo "copypaste SIM: ${DATA_PATH} en"

	cd ${LOCAL_CODE_PATH}

fi


######################################################################
if false
then
	DATA_PATH=${ROOT_PATH}/data/BytedanceSpeech/seed-tts-eval/
	torchrun ${DISTRIBUTED_ARGS} evaluation/evaluate_seedtts.py \
		--data_path ${DATA_PATH} \
		--model_name_or_path ${MODEL_NAME_OR_PATH} \
		--audio_tokenizer_path ${AUDIO_TOKENIZER_PATH} \
		--audio_tokenizer_type ${AUDIO_TOKENIZER_TYPE} \
		--flow_path ${FLOW_PATH} \
		--output_dir ${OUTPUT_DIR}/seed-tts/ \

	export ARNOLD_WORKER_GPU=${NPROC_PER_NODE}
	cd ${LOCAL_CODE_PATH}/third_party/seed-tts-eval

	bash cal_wer.sh ${DATA_PATH}/seedtts_testset/zh/meta.lst ${OUTPUT_DIR}/seed-tts/zh/ zh
	echo "copypaste WER: ${DATA_PATH} zh"
	bash cal_wer.sh ${DATA_PATH}/seedtts_testset/zh/hardcase.lst ${OUTPUT_DIR}/seed-tts/hardcase/ zh
	echo "copypaste WER: ${DATA_PATH} hardcase"
	bash cal_wer.sh ${DATA_PATH}/seedtts_testset/en/meta.lst ${OUTPUT_DIR}/seed-tts/en/ en
	echo "copypaste WER: ${DATA_PATH} en"

	cd ${LOCAL_CODE_PATH}

fi


set +x
