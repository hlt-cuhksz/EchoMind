
#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import random

import datasets

from peft import LoraConfig, LoraModel

import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers import GenerationConfig
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM

from src.feature_extraction_audio import FEATURE_EXTRACTION_MAPPING
from src.modeling_audio_encoder import AUDIO_ENCODER_MAPPING
from src.modeling_tts_lm import TTS_LM_MAPPING
from src.constants import DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN, DEFAULT_TTS_END_TOKEN
from src.instruction_dataset import load_instruction_datasets, InstructionDataCollator
from src.modeling_omnispeech import OmniSpeechModel
from src.configuration_omnispeech import OmniSpeechConfig



logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    omnispeech_model: str = field(
        default="", metadata={"help": "the path of omnispeech model"}
    )
    llm_model: str = field(
        default="Qwen/Qwen2-7B-Instruct", metadata={"help": "the path of llm model (ignored if omnispeech_model set)"}
    )
    audio_model: str = field(
        default="openai/whisper-large-v3", metadata={"help": "the path of audio model (ignored if unigpt_model set"}
    )
    tts_model: str = field(
        default="", metadata={"help": "the path of tts model (ignored if unigpt_model set"}
    )
    llm_weight: float = field(
        default=1.0, metadata={"help": "the weight for compute llm loss"}
    )
    tts_weight: float = field(
        default=1.0, metadata={"help": "the weight for compute tts loss"}
    )
    add_lora: bool = field(
        default=False, metadata={"help": "whether to add lora to adapt llm parameters"}
    )
    lora_r: int = field(
        default=16, metadata={"help": "the rank of lora module"}
    )
    lora_alpha: int = field(
        default=16, metadata={"help": "the alpha of lora module"}
    )
    lora_dropout: float = field(
        default=0.05, metadata={"help": "the dropout ratio of lora module"}
    )
    lora_target_modules: str = field(
        # default="c_attn,c_proj,w1,w2", metadata={"help": "the target modules of lora module"}
        default="q_proj,k_proj,v_proj,o_proj", metadata={"help": "the target modules of lora module"}
    )
    unfreeze_llm: bool = field(
        default=False, metadata={"help": "whether to unfreeze the llm parameters (via lora)"}
    )
    unfreeze_audio: bool = field(
        default=False, metadata={"help": "whether to unfreeze the audio encoder parameters"}
    )
    unfreeze_tts: bool = field(
        default=False, metadata={"help": "whether to unfreeze the tts lm parameters"}
    )
    unfreeze_adapter: bool = field(
        default=False, metadata={"help": "whether to unfreeze the adapter parameters"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_dirs: str = field(
        default="", metadata={"help": "directories (separated by '|') to load and save processed datasets, other data "
                                      "arguments ignored if set"}
    )
    manifest_dirs: str = field(
        default="", metadata={"help": "directories (separated by '|') to load dataset from manifest files"}
    )
    manifest_files: str = field(
        default="", metadata={"help": "manifest files (separated by '|' between datasets and then ',' between files) "
                                      "of the training manifest files"}
    )
    dataset_save_dir: str = field(
        default="", metadata={"help": "save the resulting dataset for future use"}
    )


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 4. Load tokenizer
    if model_args.omnispeech_model:
        tokenizer = AutoTokenizer.from_pretrained(model_args.omnispeech_model)
        generation_config = GenerationConfig.from_pretrained(model_args.omnispeech_model)
        tts_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_args.omnispeech_model, "tts"))
        tts_generation_config = GenerationConfig.from_pretrained(os.path.join(model_args.omnispeech_model, "tts"))
        config = OmniSpeechConfig.from_pretrained(model_args.omnispeech_model)
        audio_encoder_model_type = config.audio_encoder_config.model_type
        audio_extractor = FEATURE_EXTRACTION_MAPPING[audio_encoder_model_type].from_pretrained(os.path.join(model_args.omnispeech_model, "audio"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.llm_model)
        generation_config = GenerationConfig.from_pretrained(model_args.llm_model)
        tts_tokenizer = AutoTokenizer.from_pretrained(model_args.tts_model)
        tts_generation_config = GenerationConfig.from_pretrained(model_args.tts_model)
        audio_encoder_model_type = AutoConfig.from_pretrained(model_args.audio_model).model_type
        audio_extractor = FEATURE_EXTRACTION_MAPPING[audio_encoder_model_type].from_pretrained(model_args.audio_model)

        tokenizer.add_special_tokens(
            {
                f"additional_special_tokens": [f"{DEFAULT_AUDIO_START_TOKEN}", f"{DEFAULT_AUDIO_END_TOKEN}", 
                                               f"{DEFAULT_TTS_START_TOKEN}", f"{DEFAULT_TTS_END_TOKEN}"],
            }
        )

    ### 5. Load dataset
    dataset = load_instruction_datasets(data_args, text_tokenizer=tokenizer, tts_tokenizer=tts_tokenizer)

    # 6. Load pretrained model
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    if model_args.omnispeech_model:
        model = OmniSpeechModel.from_pretrained(model_args.omnispeech_model)
    else:
        audio_encoder_config = AutoConfig.from_pretrained(model_args.audio_model)
        if audio_encoder_config.model_type == "qwen2_audio":
            audio_encoder_config = audio_encoder_config.audio_config
        llm_config = AutoConfig.from_pretrained(model_args.llm_model)
        tts_lm_config = AutoConfig.from_pretrained(model_args.tts_model)

        config = OmniSpeechConfig(
            audio_encoder_config=audio_encoder_config.to_dict(),
            llm_config=llm_config.to_dict(),
            tts_lm_config=tts_lm_config.to_dict()
        )

        model = OmniSpeechModel(config)
        audio_encoder_model_type = config.audio_encoder_config.model_type
        model.audio_encoder_model = AUDIO_ENCODER_MAPPING[audio_encoder_model_type].from_pretrained(model_args.audio_model)
        model.llm_model = AutoModelForCausalLM.from_pretrained(model_args.llm_model)
        tts_lm_model_type = config.tts_lm_config.model_type
        model.tts_lm_model = TTS_LM_MAPPING[tts_lm_model_type].from_pretrained(model_args.tts_model)

        model.llm_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)

    model.llm_weight = model_args.llm_weight
    model.tts_weight = model_args.tts_weight

    if not model_args.unfreeze_llm:
        for name, param in model.llm_model.named_parameters():
            param.requires_grad = False
    else:
        if model_args.add_lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(","),
                lora_dropout=model_args.lora_dropout,
                bias="none"
            )
            model.add_lora(lora_config)


    if not model_args.unfreeze_audio:
        for name, param in model.audio_encoder_model.named_parameters():
            param.requires_grad = False

    if not model_args.unfreeze_tts:
        for name, param in model.tts_lm_model.named_parameters():
            param.requires_grad = False

    if not model_args.unfreeze_adapter:
        for name, param in model.audio_adapter.named_parameters():
            param.requires_grad = False
        for name, param in model.tts_lm_model.spk2tts.named_parameters():
            param.requires_grad = False
        for name, param in model.tts_lm_model.spk_ln.named_parameters():
            param.requires_grad = False
        for name, param in model.llm2tts.named_parameters():
            param.requires_grad = False
        for name, param in model.llm_ln.named_parameters():
            param.requires_grad = False

    # 6. Define data collator
    data_collator = InstructionDataCollator(
        llm_pad_id=generation_config.pad_token_id,
        tts_lm_pad_id=tts_generation_config.pad_token_id,
        sampling_rate=audio_extractor.sampling_rate,
        audio_extractor=audio_extractor,
    )


    # 7. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 8. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    results = {}
    # 9. Save tokenizer for inference load
    if is_main_process(training_args.local_rank):
        if model_args.add_lora:
            model.merge_lora()
            model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        audio_extractor.save_pretrained(os.path.join(training_args.output_dir, "audio"))
        generation_config.save_pretrained(training_args.output_dir)
        tts_tokenizer.save_pretrained(os.path.join(training_args.output_dir, "tts"))
        tts_generation_config.save_pretrained(os.path.join(training_args.output_dir, "tts"))

    return results

if __name__ == "__main__":
    main()
