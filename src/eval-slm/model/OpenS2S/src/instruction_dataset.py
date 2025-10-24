import os
import logging
from typing import Dict, List, Optional, Tuple, Union, BinaryIO
import fire
import math
from pathlib import Path

import torch.distributed as dist
import numpy as np
import torch
import datasets

from dataclasses import dataclass

from transformers import AutoTokenizer
from transformers import SequenceFeatureExtractor

try:
    from .constants import AUDIO_TOKEN_INDEX, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN, DEFAULT_TTS_END_TOKEN, IGNORE_INDEX
    from .utils import get_waveform, collate_tokens
except:
    from constants import AUDIO_TOKEN_INDEX, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN, DEFAULT_TTS_END_TOKEN, IGNORE_INDEX
    from utils import get_waveform, collate_tokens


logger = logging.getLogger(__name__)


def tokenize_qwen3_str(tokenizer, role="", content="", add_bos=False, add_eos=False, enable_thinking=False):
    tokens = []
    if add_bos:
        tokens += tokenizer.encode("<|im_start|>")
    if role:
        tokens += tokenizer.encode(role) + tokenizer.encode("\n")
        if role == "assistant" and not enable_thinking:
            tokens += tokenizer.encode("<think></think>") # no reasoning
    if content:
        tokens += tokenizer.encode(content)
    if add_eos:
        tokens += tokenizer.encode("<|im_end|>\n")
    return tokens


def process_dataset(
    batch,
    text_tokenizer,
    tts_tokenizer=None,
    enable_thinking=False,
    check_audio=True,
):
    tokenize_func = tokenize_qwen3_str  ## can modified here
    system_prompt = "You are a helpful assistant."

    input_ids = []
    token_types = [] # 0 for text and audio; 1 for umm audio output
    labels = []
    audio_paths = []
    speech_units = []
    spk_emb_paths = []

    if system_prompt is not None:
        new_input_ids = tokenize_func(text_tokenizer, role="system", content=system_prompt, add_bos=True, add_eos=True)
        input_ids += new_input_ids
        token_types += [0] * len(new_input_ids)
        labels += [IGNORE_INDEX] * len(new_input_ids)

    messages = batch["messages"]

    for message in messages:
        role = message["role"]
        content = message["content"]

        new_input_ids = tokenize_func(text_tokenizer, role=role, content="", add_bos=True, add_eos=False, enable_thinking=enable_thinking)
        input_ids += new_input_ids
        token_types += [0] * len(new_input_ids)
        labels += [IGNORE_INDEX] * len(new_input_ids)

        if isinstance(content, list):
            for item in content:
                if item["audio"]:
                    ### must audio input
                    assert role == "user"
                    new_input_ids = tokenize_func(text_tokenizer, role="", content=DEFAULT_AUDIO_START_TOKEN, add_bos=False, add_eos=False)
                    input_ids += new_input_ids
                    token_types += [0] * len(new_input_ids)
                    labels += [IGNORE_INDEX] * len(new_input_ids)

                    new_input_ids = [AUDIO_TOKEN_INDEX]
                    input_ids += new_input_ids
                    token_types += [0] * len(new_input_ids)
                    labels += [IGNORE_INDEX] * len(new_input_ids)

                    audio_paths.append(item["audio"])

                    new_input_ids = tokenize_func(text_tokenizer, role="", content=DEFAULT_AUDIO_END_TOKEN, add_bos=False, add_eos=False)
                    input_ids += new_input_ids
                    token_types += [0] * len(new_input_ids)
                    labels += [IGNORE_INDEX] * len(new_input_ids)
                elif item["speech_units"]:
                    ### must audio output
                    assert role == "assistant"
                    assert item["text"]
                    assert tts_tokenizer is not None
                    text = item["text"]
                    new_input_ids = tokenize_func(
                        text_tokenizer,
                        role="",
                        content=f"{DEFAULT_TTS_START_TOKEN}{text}{DEFAULT_TTS_END_TOKEN}",
                        add_bos=False,
                        add_eos=False
                    )
                    input_ids += new_input_ids
                    token_types += [1] * len(new_input_ids)
                    labels += new_input_ids

                    speech_unit = item["speech_units"]
                    speech_units.append(tts_tokenizer.encode(speech_unit) + [tts_tokenizer.eos_token_id])
                    if "spk_emb" in item:
                        spk_emb_paths.append(item["spk_emb"])
                    else:
                        spk_emb_paths.append("")
                elif item["text"]:
                    text = item["text"]
                    new_input_ids = tokenize_func(text_tokenizer, role="", content=text, add_bos=False, add_eos=False)
                    input_ids += new_input_ids
                    token_types += [0] * len(new_input_ids)
                    if role == "user":
                        labels += [IGNORE_INDEX] * len(new_input_ids)
                    elif role == "assistant":
                        labels += new_input_ids
                    else:
                        raise NotImplementedError
                else:
                    continue
            new_input_ids = tokenize_func(text_tokenizer, role="", content="", add_bos=False, add_eos=True)
            input_ids += new_input_ids
            token_types += [0] * len(new_input_ids)
            if role == "user":
                labels += [IGNORE_INDEX] * len(new_input_ids)
            elif role == "assistant":
                labels += new_input_ids
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    to_keep = True
    if check_audio:
        try:
            for audio_path in audio_paths:
                waveform = get_waveform(audio_path)
                if waveform.shape[0] < 8000:
                    to_keep = False
                    break
        except:
            print(audio_paths)
            to_keep = False

    batch["to_keep"] = to_keep

    ### here is a hook to ensure consistent data type
    if not audio_paths:
        audio_paths.append("hook")
    if not speech_units:
        speech_units.append([0])
    if not spk_emb_paths:
        spk_emb_paths.append("hook")

    
    batch["input_ids"] = input_ids
    batch["token_types"] = token_types
    batch["labels"] = labels
    batch["audio_paths"] = audio_paths
    batch["speech_units"] = speech_units
    batch["spk_emb_paths"] = spk_emb_paths

    return batch


def load_instruction_dataset(
    manifest_dir="",
    manifest_files="",
    text_tokenizer=None,
    tts_tokenizer=None,
    enable_thinking=False,
    num_proc=32,
):
    if not manifest_files:
        logger.warning(f"loading processed dataset from {manifest_dir}")
        dataset = datasets.load_from_disk(manifest_dir)
        return dataset
    
    logger.warning(f"load dataset from scratch from {manifest_dir}/{manifest_files}")
    
    manifest_files_list = manifest_files.split(",")

    raw_dataset = datasets.load_dataset(
        manifest_dir, data_files=manifest_files_list, split="train", streaming=False
    )

    dataset = raw_dataset.map(
        process_dataset,
        fn_kwargs={
            "text_tokenizer": text_tokenizer,
            "tts_tokenizer": tts_tokenizer,
            "enable_thinking": enable_thinking,
        },
        load_from_cache_file=False,
        remove_columns=raw_dataset.column_names,
        num_proc=num_proc,
    )

    def to_keep(flag):
        return flag

    dataset = dataset.filter(
        to_keep,
        input_columns=["to_keep"]
    )

    return dataset


def load_instruction_datasets(data_args, text_tokenizer=None, tts_tokenizer=None, num_proc=32):
    if os.path.exists(data_args.dataset_save_dir) and os.listdir(data_args.dataset_save_dir):
        logger.warning(f"loading processed dataset from {data_args.dataset_save_dir}")
        dataset = datasets.load_from_disk(data_args.dataset_save_dir)
        return dataset

    manifest_keys = ["manifest_dirs", "manifest_files"]
    if data_args.dataset_dirs:
        dataset_dirs = data_args.dataset_dirs.split("|")
        all_datasets = [load_instruction_dataset(manifest_dir=dataset_dir) for dataset_dir in dataset_dirs]
    else:
        manifest_values = [(getattr(data_args, key)).split("|") for key in manifest_keys]
        num_datasets = len(manifest_values[0])
        if num_datasets == 0:
            raise ValueError("no datasets specified")
        for i, key in enumerate(manifest_keys):
            if len(manifest_values[i]) != num_datasets:
                raise ValueError(f"unexpected number of {key} in {data_args}")
        all_datasets = [load_instruction_dataset(manifest_dir=manifest_values[0][i],
                                                 manifest_files=manifest_values[1][i],
                                                 text_tokenizer=text_tokenizer,
                                                 tts_tokenizer=tts_tokenizer,
                                                 num_proc=num_proc)
                        for i in range(num_datasets)]
    if len(all_datasets) == 1:
        dataset = all_datasets[0]
    else:
        dataset = datasets.concatenate_datasets(all_datasets)

    
    if data_args.dataset_save_dir and (not dist.is_initialized() or dist.get_rank() == 0):
        dataset.save_to_disk(data_args.dataset_save_dir)

    return dataset


@dataclass
class InstructionDataCollator:
    """
    Data collator that will dynamically pad the inputs received.
    """
    llm_pad_id: int = 0
    tts_lm_pad_id: int = 0
    sampling_rate: int = 16000
    audio_extractor: SequenceFeatureExtractor = None

    def __call__(self, samples: List[Dict]):
        input_ids = [sample["input_ids"] for sample in samples]
        attention_mask = [[1] * len(sample["input_ids"]) for sample in samples]
        token_types = [sample["token_types"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        speech_units = [item for sample in samples for item in sample["speech_units"] if item != [0]]
        speech_units_mask = [[1] * len(item) for sample in samples for item in sample["speech_units"] if item != [0]]

        input_ids = collate_tokens(input_ids, self.llm_pad_id)
        attention_mask = collate_tokens(attention_mask, 0)
        token_types = collate_tokens(token_types, IGNORE_INDEX)
        labels = collate_tokens(labels, IGNORE_INDEX)

        if speech_units != []:
            speech_units = collate_tokens(speech_units, self.tts_lm_pad_id)
            speech_units_mask = collate_tokens(speech_units_mask, 0)
        else:
            speech_units, speech_units_mask = None, None

        raw_speech = [
            get_waveform(item, output_sample_rate=self.sampling_rate)
            for sample in samples for item in sample["audio_paths"] if item != "hook"
        ]
        if len(raw_speech) == 0:
            speech_values = None
            speech_mask = None
        else:
            speech_inputs = self.audio_extractor(
                raw_speech, 
                sampling_rate=self.sampling_rate, 
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_values = speech_inputs.input_features
            speech_mask = speech_inputs.attention_mask

        spk_emb_paths = [item for sample in samples for item in sample["spk_emb_paths"] if item != "hook"]
        spk_embs = [
            np.load(item) if item != "" else None
            for item in spk_emb_paths
        ]
        spk_embs = [
            torch.from_numpy(i) if i is not None else torch.zeros(1,1,512)
            for i in spk_embs
        ]
        if spk_embs != []:
            spk_embs = torch.cat(spk_embs, dim=0) # B x 1 x 512
        else:
            spk_embs = None

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_types": token_types,
            "labels": labels,
            "speech_values": speech_values,
            "speech_mask": speech_mask,
            "speech_units": speech_units,
            "speech_units_mask": speech_units_mask,
            "spk_embs": spk_embs,
        }


def offline_process(
    dataroot="",
    manifest_files="",
    llm_path="",
    tts_path="",
    enable_thinking=False,
    save_dir="",
    num_proc=32,
):
    text_tokenizer = AutoTokenizer.from_pretrained(llm_path)
    text_tokenizer.add_special_tokens(
        {
            f"additional_special_tokens": [f"{DEFAULT_AUDIO_START_TOKEN}", f"{DEFAULT_AUDIO_END_TOKEN}"],
        }
    )
    text_tokenizer.add_special_tokens(
        {
            f"additional_special_tokens": [f"{DEFAULT_TTS_START_TOKEN}", f"{DEFAULT_TTS_END_TOKEN}"],
        }
    )
    if tts_path:
        tts_tokenizer = AutoTokenizer.from_pretrained(tts_path)
    else:
        tts_tokenizer = None

    dataset = load_instruction_dataset(
        dataroot,
        manifest_files,
        text_tokenizer,
        tts_tokenizer,
        enable_thinking,
        num_proc,
    )
    print(len(dataset))
    for key in dataset[0].keys():
        print(key, dataset[0][key])
        if key == "input_ids":
            input_ids = [item if item != AUDIO_TOKEN_INDEX else 0 for item in dataset[0]["input_ids"]]
            print("input", text_tokenizer.decode(input_ids))
    
    if save_dir:
        dataset.save_to_disk(save_dir)


if __name__ == "__main__":
    fire.Fire({
        "offline": offline_process,
    })
