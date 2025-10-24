import json
import logging
import math
import os
import pdb
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from .dataset_base import BaseDataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class CosyVoice2Dataset(BaseDataset):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.default_system_message = "You are a helpful AI assistant."
        self.default_system_message = None

        self.ret = defaultdict(dict)
        self.is_cat = True

        if self.cross_dataset_joint:
            for i in range(2):
                self.maybe_init_ret(f"default_{i}")

    def maybe_init_ret(self, source, force=False):
        if source not in self.ret or force:
            self.ret[source] = {}

            self.ret[source]["tokens"] = []
            self.ret[source]["labels"] = []
            self.ret[source]["actual_seq_len"] = []

            if self.create_position_ids:
                self.ret[source]["position_ids"] = []

            if self.create_attention_mask:
                self.ret[source]["attention_mask"] = []

            if self.create_attention_mask_2d:
                self.ret[source]["attention_mask_2d"] = torch.tril(
                    torch.ones(
                        (1, self.max_padding_length, self.max_padding_length), dtype=torch.bool
                    )
                )
        return len(self.ret[source]["tokens"]) == 0

    def get_max_min_ret_length(self):
        max_ret_lengh = 0
        min_ret_lengh = self.max_padding_length + 1

        max_ret_key = None
        min_ret_key = None

        for k, v in self.ret.items():
            cur_length = len(v["tokens"])

            if cur_length > max_ret_lengh:
                max_ret_lengh = cur_length
                max_ret_key = k

            if cur_length < min_ret_lengh:
                min_ret_lengh = cur_length
                min_ret_key = k

        return max_ret_lengh, max_ret_key, min_ret_lengh, min_ret_key

    def add_ret(self, ret, source):
        cur_length = len(ret["input_ids"])
        cur_image_length = len(ret["images"])

        all_length = len(self.ret[source]["tokens"])
        if "images" in self.ret[source]:
            all_image_length = len(self.ret[source]["images"])
        else:
            all_image_length = 0

        if cur_image_length > 0:
            if all_image_length > 0:
                self.ret[source]["images"] = torch.cat(
                    [self.ret[source]["images"], ret["images"]], dim=0
                )
                ret["image_indices"][1, :, :] += all_length
                self.ret[source]["image_indices"] = torch.cat(
                    [self.ret[source]["image_indices"], ret["image_indices"]], dim=1
                )
            else:
                self.ret[source]["images"] = ret["images"]
                self.ret[source]["image_indices"] = ret["image_indices"]

        if self.create_attention_mask:
            self.ret[source]["attention_mask"] += ret["attention_mask"]

        if self.create_attention_mask_2d:
            self.ret[source]["attention_mask_2d"][:, all_length:, :all_length] = 0

        if self.create_position_ids:
            self.ret[source]["position_ids"] += list(range(cur_length))

        self.ret[source]["tokens"] += ret["input_ids"]
        self.ret[source]["labels"] += ret["labels"]
        self.ret[source]["actual_seq_len"] += [all_length + cur_length]

    def process_ret(self, to_ret):
        if "tokens" in to_ret and len(to_ret["tokens"]) > 0:
            pass
        else:
            return to_ret

        if self.create_position_ids:
            if self.reset_position_ids:
                pass
            else:
                to_ret["position_ids"] = list(range(len(to_ret["tokens"])))

        if self.create_attention_mask_2d:
            if self.reset_attention_mask:
                pass
            else:
                to_ret["attention_mask_2d"] = torch.tril(
                    torch.ones(
                        (1, self.max_padding_length, self.max_padding_length), dtype=torch.bool
                    )
                )

        if self.shift_token:
            to_ret["tokens"] = to_ret["tokens"][:-1]
            to_ret["labels"] = to_ret["labels"][1:]
            to_ret["actual_seq_len"][-1] -= 1
            if self.create_position_ids:
                to_ret["position_ids"] = to_ret["position_ids"][:-1]
            if self.create_attention_mask:
                to_ret["attention_mask"] = to_ret["attention_mask"][:-1]

            if self.create_attention_mask_2d:
                to_ret["attention_mask_2d"][:, :, -1] = 0
                to_ret["attention_mask_2d"][:, -1, :] = 0

        assert len(to_ret["tokens"]) == len(
            to_ret["labels"]
        ), f"{len(to_ret['tokens'])} {len(to_ret['labels'])}"
        if not self.variable_length and self.max_padding_length > len(to_ret["tokens"]):
            to_ret["tokens"] += [self.tokenizer.pad_token_id] * (
                self.max_padding_length - len(to_ret["tokens"])
            )
            to_ret["labels"] += [IGNORE_TOKEN_ID] * (
                self.max_padding_length - len(to_ret["labels"])
            )
            to_ret["actual_seq_len"][-1] = self.max_padding_length
            if self.create_position_ids:
                # to_ret["position_ids"] += to_ret["position_ids"][-1:] * (
                #     self.max_padding_length - len(to_ret["position_ids"])
                # )
                to_ret["position_ids"] += list(
                    range(to_ret["position_ids"][-1] + 1, self.max_padding_length)
                )
            if self.create_attention_mask:
                to_ret["attention_mask"] += [0] * (
                    self.max_padding_length - len(to_ret["attention_mask"])
                )

        to_ret["tokens"] = to_ret["tokens"][: self.max_padding_length]
        to_ret["labels"] = to_ret["labels"][: self.max_padding_length]
        to_ret["actual_seq_len"][-1] = self.max_padding_length
        if self.create_position_ids:
            to_ret["position_ids"] = to_ret["position_ids"][: self.max_padding_length]
        if self.create_attention_mask:
            to_ret["attention_mask"] = to_ret["attention_mask"][: self.max_padding_length]

        to_ret["tokens"] = torch.tensor(to_ret["tokens"], dtype=torch.int64)
        to_ret["labels"] = torch.tensor(to_ret["labels"], dtype=torch.int64)
        to_ret["actual_seq_len"] = torch.tensor(to_ret["actual_seq_len"], dtype=torch.int64)
        if self.create_position_ids:
            to_ret["position_ids"] = torch.tensor(to_ret["position_ids"], dtype=torch.int64)
        if self.create_attention_mask:
            to_ret["attention_mask"] = torch.tensor(to_ret["attention_mask"], dtype=torch.int64)

        if self.create_attention_mask_2d:
            attention_mask_2d = to_ret.pop("attention_mask_2d")
            attention_mask_2d = attention_mask_2d.masked_fill(
                (to_ret["attention_mask"] < 0.5).view(1, 1, self.max_padding_length), value=0
            )
            attention_mask_2d = attention_mask_2d < 0.5

            to_ret["attention_mask"] = attention_mask_2d

        if self.create_loss_mask:
            loss_mask = torch.where(to_ret["labels"] == IGNORE_TOKEN_ID, 0, 1)
            to_ret["loss_mask"] = loss_mask.to(torch.float32)

        if not self.reset_position_ids and not self.reset_attention_mask:
            to_ret.pop("actual_seq_len")

        to_ret["input_ids"] = to_ret["tokens"]

        # print("to_ret[tokens]", to_ret["tokens"])
        # print("to_ret[labels]", to_ret["labels"])

        return to_ret

    def is_skip(self):
        if self.processed_samples < self.skip_samples:
            if self.processed_samples % 1e3 == 0:
                print(
                    f"processed_samples {self.processed_samples} skip_samples {self.skip_samples}"
                )
            return True

    def show_statistic(self):
        log_interval = 10000
        if self.max_padding_length >= 2**17:
            log_interval = 500
        if self.max_padding_length >= 2**20:
            log_interval = 100

        if self.unjoint_samples % log_interval == 0:
            print(
                f"processed_samples {self.processed_samples} unjoint_samples {self.unjoint_samples} joint_samples {self.joint_samples} {[len(v['tokens']) for _, v in self.ret.items()]}",
                flush=True,
            )

        return False

    def __getitem__(self, index):

        index = index % self.__len__()

        if "audio" in self.processor and self.processor["audio"] is not None:
            self.processor["audio"].audio_tokenizer.load_model()

        while True:
            # if True:
            try:
                self.processed_samples += 1
                if self.is_skip():
                    return {}

                sample = self.raw_data[index]

                if self.cross_dataset_joint:
                    is_empty = False
                    (
                        max_ret_lengh,
                        max_ret_key,
                        min_ret_lengh,
                        min_ret_key,
                    ) = self.get_max_min_ret_length()
                else:
                    source = sample["source"]
                    is_empty = self.maybe_init_ret(source)

                    max_ret_lengh = min_ret_lengh = len(self.ret[source]["tokens"])
                    max_ret_key = min_ret_key = source

                is_begin = is_empty or self.reset_position_ids or self.reset_attention_mask

                ret = preprocess(
                    sample,
                    self.tokenizer,
                    self.image_token_length,
                    default_system_message=self.default_system_message,
                    processor=self.processor,
                    is_begin=is_begin,
                    max_num_frame=self.max_num_frame,
                    max_fps=self.max_fps,
                )

                if ret is None:
                    return {}

                cur_length = len(ret["input_ids"])

                if cur_length > self.max_padding_length:
                    return {}

                self.unjoint_samples += 1

                if not self.dataset_joint:
                    to_ret = self.ret.pop(max_ret_key)

                    self.maybe_init_ret(max_ret_key, force=True)
                    self.add_ret(ret, max_ret_key)

                elif min_ret_lengh + cur_length > self.max_padding_length:
                    to_ret = self.ret.pop(max_ret_key)
                    self.joint_samples += 1

                    self.maybe_init_ret(max_ret_key, force=True)
                    self.add_ret(ret, max_ret_key)

                else:
                    to_ret = {}
                    self.add_ret(ret, min_ret_key)

                to_ret = self.process_ret(to_ret)

                self.show_statistic()
                return to_ret

            except Exception as error:
                try:
                    with open(os.path.join(self.output_dir, "data_error.log"), "a") as f:
                        print("-" * 100, file=f)
                        print(traceback.format_exc(), file=f)
                        print(self.raw_data[index], file=f)
                except Exception as error:
                    print(error)
                return {}


def preprocess(
    sample,
    tokenizer: transformers.PreTrainedTokenizer,
    image_token_length: int,
    default_system_message: str = "You are a helpful assistant.",
    processor=None,
    is_begin: bool = True,
    max_num_frame: int = 8,
    max_fps: int = 1,
) -> Dict:

    from ..constants import (
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        VID_START_TOKEN,
        VID_END_TOKEN,
        VID_CONTEXT_TOKEN,
        PATCH_START_TOKEN,
        PATCH_END_TOKEN,
        PATCH_CONTEXT_TOKEN,
        AUD_START_TOKEN,
        AUD_END_TOKEN,
        IMG_TAG_TOKEN,
        VID_TAG_TOKEN,
        AUD_TAG_TOKEN,
    )

    human_roles = ["user", "human"]
    gpt_roles = ["assistant", "gpt"]
    system_roles = ["system"]

    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids

    AUD_TAG_ID = AUD_TAG_ID[0]

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    input_ids, targets = [], []
    images = []
    image_indices = []

    messages = []
    if "conversations" in sample:
        messages = sample["conversations"]
    if len(messages) == 0 and "messages" in sample:
        messages = sample["messages"]

    # ----------------------------------------------------------------
    # audio
    if has_audio(sample):
        audio_tokens_list = [processor["audio"].process_audios(x) for x in sample["audios"]]
        audio_tokens_list = ["".join(f"<|audio_{i}|>" for i in x) for x in audio_tokens_list]

        audio_idx = 0
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            while AUD_TAG_TOKEN in content:
                content = content.replace(
                    AUD_TAG_TOKEN,
                    f"{audio_tokens_list[audio_idx]}",
                    1,
                )
                audio_idx += 1

            sentence["content"] = content

        audio_idx = 0
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            while "<audio>" in content:
                content = content.replace("<audio>", f"{audio_tokens_list[audio_idx]}", 1)
                audio_idx += 1

            sentence["content"] = content

    # ----------------------------------------------------------------
    # text
    for j, sentence in enumerate(messages):
        role = sentence["role"]
        content = sentence["content"]

        if role in human_roles:
            text = content.replace("Convert the text to speech.\n", "")
            text = text.strip()

        elif role in gpt_roles:
            audio = content

        else:
            raise NotImplementedError

    text_token = tokenizer(text, add_special_tokens=False).input_ids
    speech_token = tokenizer(audio, add_special_tokens=False).input_ids

    text_token_len = len(text_token)
    speech_token_len = len(speech_token)

    mix_ratio = [5, 15]

    sos_eos_id = 151663
    task_id_id = 151664
    # speech_token_size = 6561
    # speech_token_offset = 151665

    end_token_id = 151665 + 6561
    fill_token_id = 151665 + 6561 + 2

    # bistream sequence
    if random.random() < 0.5 and speech_token_len / text_token_len > mix_ratio[1] / mix_ratio[0]:
        # if speech_token_len / text_token_len > mix_ratio[1] / mix_ratio[0]:

        targets.append(IGNORE_TOKEN_ID)
        input_ids.append(sos_eos_id)

        for j in range(math.ceil((text_token_len + 1) / mix_ratio[0])):

            this_text_token = text_token[j * mix_ratio[0] : (j + 1) * mix_ratio[0]]
            this_speech_token = speech_token[j * mix_ratio[1] : (j + 1) * mix_ratio[1]]

            if len(this_text_token) == mix_ratio[0]:
                assert len(this_speech_token) == mix_ratio[1]
                targets += (
                    [IGNORE_TOKEN_ID] * (mix_ratio[0] - 1) + this_speech_token + [fill_token_id]
                )
                input_ids += this_text_token + this_speech_token

            else:
                this_speech_token = speech_token[j * mix_ratio[1] :]
                targets += (
                    [IGNORE_TOKEN_ID] * len(this_text_token) + this_speech_token + [end_token_id]
                )
                input_ids += this_text_token + [task_id_id] + this_speech_token

    # unistream sequence
    else:
        targets = [IGNORE_TOKEN_ID] * (1 + text_token_len) + speech_token + [end_token_id]
        input_ids = [sos_eos_id] + text_token + [task_id_id] + speech_token

    # shift
    # targets = [IGNORE_TOKEN_ID] + targets
    # input_ids = input_ids + [end_token_id]

    attention_mask = [1] * len(input_ids)

    # print("sample", sample, flush=True)
    # print("input_ids", input_ids, flush=True)
    # print("targets", targets[:100], flush=True)
    # print("images", [xx.shape for x in images for xx in x], flush=True)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
        images=images,
        image_indices=image_indices,
    )


def has_audio(sample):
    # audio
    if (
        "audios" in sample
        and isinstance(sample["audios"], list)
        and None not in sample["audios"]
        and len(sample["audios"])
    ):
        return True
    return False
