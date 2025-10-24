#!/bin/bash

set -e
set -x

SEQ_LENGTH="$1"
if [ -z "$SEQ_LENGTH" ]
then
    SEQ_LENGTH=32768
fi

timestamp="$2"
if [ -z "$timestamp" ]
then
    timestamp=`date +'%Y%m%d_%H'`0000
fi

######################################################################
export ROOT_PATH=/data/
export CODE_PATH=${ROOT_PATH}/VITA-Audio/

export LOCAL_ROOT_PATH=/data_local/
export LOCAL_CODE_PATH=${LOCAL_ROOT_PATH}/VITA-Audio/
mkdir -p ${LOCAL_ROOT_PATH}
mkdir -p ${LOCAL_CODE_PATH}

apt update
apt install -y rsync
rsync -a --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/

cd ${LOCAL_CODE_PATH}
rm -fr datasets
ln -s ${ROOT_PATH}/data datasets

######################################################################
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source ${LOCAL_CODE_PATH}/scripts/set_env_ds_gpu.sh
pip3 install transformers==4.48.3
#pip3 install --no-index --find-links=/data/software/ transformers==4.48.3

######################################################################
OUTPUT_DIR=${ROOT_PATH}/output/LM/"$0"/${timestamp}/

mkdir -p ${OUTPUT_DIR}
rsync -avh $0 ${OUTPUT_DIR}

export HF_HOME="${ROOT_PATH}/data/HF_HOME_node${INDEX}/"
mkdir -p ${HF_HOME}

export TRITON_CACHE_DIR=${LOCAL_CODE_PATH}

export PYTHONPATH=$PYTHONPATH:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice/third_party/Matcha-TTS/

######################################################################
LOG=${OUTPUT_DIR}/log_node${INDEX}.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

echo ${@}

######################################################################
DATA_PATH=${LOCAL_CODE_PATH}/configs/sts_finetune_stage2.yaml

MODEL_NAME_OR_PATH=${ROOT_PATH}/output/LM/scripts/deepspeed/sts_qwen25/finetune_sensevoice_glm4voice_mtp10_stage1.sh/20250421_180624/

AUDIO_TOKENIZER_PATH=${ROOT_PATH}/models/THUDM/glm-4-voice-tokenizer

rsync -avh ${DATA_PATH} ${OUTPUT_DIR}

######################################################################
DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS tools/finetune_sts_v4_48_3.py \
    --log_level "info" \
    --do_train \
    --overwrite_output_dir \
    --config_name ${LOCAL_CODE_PATH}/VITA-Audio/models/qwen2_mtp_sensevoice_v4_48_3/config_7B_mtp10.json \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_tokenizer_path $AUDIO_TOKENIZER_PATH \
    --audio_tokenizer_type "sensevoice_glm4voice" \
    --dataset_name $DATA_PATH \
    --bf16 True \
    --tf32 True \
    --torch_dtype bfloat16 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --max_steps 4000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 2 \
    --learning_rate 5.00e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length ${SEQ_LENGTH} \
    --gradient_checkpointing True \
    --deepspeed ${LOCAL_CODE_PATH}/scripts/deepspeed/ds_config_zero2_no_optimizer.json \
    --trust_remote_code False \
    --ddp_timeout 7200 \
    --ddp_backend ${DISTRIBUTED_BACKEND} \
    --attn_implementation flash_attention_2 \
    --seed 42 \
    --data_seed 42 \
    --reset_attention_mask \
    --reset_position_ids \
    --create_attention_mask false \
    --create_attention_mask_2d false \
    --dataloader_num_workers 2 \
    --mtp_model_lr_mult 1.00e1 \
    --audio-model-freeze \
    --text-audio-interval-ratio 1 10 4 10 \

    #--language-model-freeze \
    #--dataset_joint false \
    #--variable_length true \
    #--tokenizer_name_or_path Qwen2Tokenizer \

    #--bf16 True \
    #--fp16 True \
    #--tf32 True \

set +x
