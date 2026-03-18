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
rsync -avh -P ${CODE_PATH}/Kimi-Audio-Evalkit/ /data/Kimi-Audio-Evalkit/

cd /data/Kimi-Audio-Evalkit/




######################################################################
if true
#if false
then

    bash run_audio.sh \
        --model VITA-Audio \
        --data "LibriSpeech AISHELL-1 AISHELL-2 WenetSpeech Fleurs-en Fleurs-zh" \
        --work-dir ${OUTPUT_DIR} \

fi

if true
#if false
then

    bash run_audio.sh \
        --model VITA-Audio \
        --data "mmsu openbookqa sd-qa advbench alpacaeval_full commoneval ifeval OpenAudioBench" \
        --work-dir ${OUTPUT_DIR} \
        --skip-eval 

    export OPENAI_API_KEY=""

    bash run_audio.sh \
        --model VITA-Audio \
        --data "mmsu openbookqa sd-qa advbench alpacaeval_full commoneval ifeval OpenAudioBench" \
        --work-dir ${OUTPUT_DIR} \
        --reeval 

fi

set +x
