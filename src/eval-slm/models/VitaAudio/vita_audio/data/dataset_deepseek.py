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


class DeepSeekDataset(BaseDataset):
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
        cur_audio_length = len(ret["audios"])

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

        if "audios" in self.ret[source]:
            all_audio_length = len(self.ret[source]["audios"])
        else:
            all_audio_length = 0

        if cur_audio_length > 0:
            if all_audio_length > 0:
                # self.ret[source]["audios"] = torch.cat(
                #     [self.ret[source]["audios"], ret["audios"]], dim=0
                # )
                # ret["audio_indices"][1, :, :] += all_length
                # self.ret[source]["audio_indices"] = torch.cat(
                #     [self.ret[source]["audio_indices"], ret["audio_indices"]], dim=1
                # )
                self.ret[source]["audios"].extend(ret["audios"])
                for audio_indice in ret["audio_indices"]:
                    audio_indice[1, :, :] += all_length
                self.ret[source]["audio_indices"].extend(ret["audio_indices"])
            else:
                self.ret[source]["audios"] = ret["audios"]
                self.ret[source]["audio_indices"] = ret["audio_indices"]

            # print(self.ret[source]["audios"])

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

        self.processor["audio"].load_model()

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

    # <|im_start|>system
    # You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # Hello, how are you?<|im_end|>
    # <|im_start|>assistantI'm doing great. How can I help you today?<|im_end|>
    # <|im_start|>user
    # I'd like to show off how chat templating works!<|im_end|>

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
        AUD_CONTEXT_TOKEN,
    )

    human_roles = ["user", "human"]
    gpt_roles = ["assistant", "gpt"]
    system_roles = ["system"]

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
    AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids

    IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
    VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    assert len(VID_CONTEXT_ID) == 1
    assert len(VID_START_ID) == 1
    assert len(VID_END_ID) == 1

    assert len(PATCH_CONTEXT_ID) == 1
    assert len(PATCH_START_ID) == 1
    assert len(PATCH_END_ID) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    VID_CONTEXT_ID = VID_CONTEXT_ID[0]
    VID_START_ID = VID_START_ID[0]
    VID_END_ID = VID_END_ID[0]

    PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
    PATCH_START_ID = PATCH_START_ID[0]
    PATCH_END_ID = PATCH_END_ID[0]

    AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
    AUD_START_ID = AUD_START_ID[0]
    AUD_END_ID = AUD_END_ID[0]

    IMG_TAG_ID = IMG_TAG_ID[0]
    VID_TAG_ID = VID_TAG_ID[0]
    AUD_TAG_ID = AUD_TAG_ID[0]

    BOS_ID = tokenizer.bos_token_id
    EOS_ID = tokenizer.eos_token_id

    IM_START = "<｜begin▁of▁sentence｜>"
    IM_END = "<｜end▁of▁sentence｜>"
    USER = "<｜User｜>"
    ASSISTANT = "<｜Assistant｜>"

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids
    IM_START_IDS = tokenizer(IM_START, add_special_tokens=False).input_ids
    IM_END_IDS = tokenizer(IM_END, add_special_tokens=False).input_ids
    USER_IDS = tokenizer(USER, add_special_tokens=False).input_ids
    ASSISTANT_IDS = tokenizer(ASSISTANT, add_special_tokens=False).input_ids

    assert len(USER_IDS) == 1, USER_IDS
    assert len(ASSISTANT_IDS) == 1, ASSISTANT_IDS
    assert len(IM_END_IDS) == 1, IM_END_IDS
    assert len(IM_START_IDS) == 1, IM_START_IDS

    input_ids, targets = [], []
    images = []
    image_indices = []
    audios = []
    audio_indices = []

    messages = []
    if "conversations" in sample:
        messages = sample["conversations"]
    if len(messages) == 0 and "messages" in sample:
        messages = sample["messages"]

    # ----------------------------------------------------------------
    # add text to TTS
    if True:
        add_text = None
        # add_audio = None
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            role = sentence["role"]
            if role == "user":
                if "Convert the text to speech." in content:
                    add_text = content.replace("Convert the text to speech.\n", "")
                    add_text = add_text.strip()

                # if "Convert the speech to text." in content:
                #     add_audio = sample["audios"][-1]

            if role == "assistant" and add_text is not None:
                sentence["content"] = add_text + content

            # if role == "assistant" and add_audio is not None:
            #     sentence["content"] = content + "\n<audio>"
            #     sample["audios"].append(add_audio)

    # ----------------------------------------------------------------
    # system
    has_system = False
    if is_begin:
        if messages[0]["role"] == "system":
            has_system = True
        else:
            has_system = False

        if (
            not has_system
            and default_system_message is not None
            and len(default_system_message) > 0
        ):
            messages = [{"role": "system", "content": default_system_message}] + messages
            has_system = True

    # ----------------------------------------------------------------
    # audio
    if has_audio(sample) and processor["audio"].is_discrete:
        unused_audio_idxs = list(range(len(sample["audios"])))
        audio_tokens_list = [
            processor["audio"].process_audios(x, is_discrete=True) for x in sample["audios"]
        ]
        audio_tokens_list = ["".join(f"<|audio_{i}|>" for i in x) for x in audio_tokens_list]

        audio_idx = 0
        for j, sentence in enumerate(messages):
            content = sentence["content"]
            role = sentence["role"]
            # whether apply discrete tokenize to this role
            if processor["audio"].apply_to_role(role, is_discrete=True):
                while AUD_TAG_TOKEN in content:
                    content = content.replace(
                        AUD_TAG_TOKEN,
                        f"{AUD_START_TOKEN}{audio_tokens_list[audio_idx]}{AUD_END_TOKEN}",
                        1,
                    )
                    unused_audio_idxs.remove(audio_idx)
                    audio_idx += 1
            else:
                audio_idx += content.count(AUD_TAG_TOKEN)

            sentence["content"] = content

    # ----------------------------------------------------------------
    input_ids += IM_START_IDS
    targets = [IGNORE_TOKEN_ID] * len(IM_START_IDS)
    # text
    for j, sentence in enumerate(messages):
        role = sentence["role"]
        content = sentence["content"]

        if role in human_roles:
            _input_id = USER_IDS + tokenizer(content, add_special_tokens=False).input_ids
            _target = [IGNORE_TOKEN_ID] * len(_input_id)

        elif role in gpt_roles:
            content_input_id = tokenizer(content, add_special_tokens=False).input_ids
            if processor["audio"].audio_tokenizer is not None:
                content_input_id = text_audio_interval(
                    content_input_id,
                    AUD_START_ID,
                    AUD_END_ID,
                    processor["audio"].text_audio_interval_ratio,
                )

            _input_id = ASSISTANT_IDS + content_input_id + IM_END_IDS
            _target = [IGNORE_TOKEN_ID] * len(ASSISTANT_IDS) + content_input_id + IM_END_IDS

        elif role in system_roles:
            if "神农" in content and "腾讯" in content:
                continue
            _input_id = tokenizer(content, add_special_tokens=False).input_ids
            _target = [IGNORE_TOKEN_ID] * len(_input_id)

        else:
            raise NotImplementedError

        # print(f"_input_id {_input_id}")
        input_ids += _input_id
        targets += _target

    # ----------------------------------------------------------------
    # image
    if has_image(sample):
        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]
        assert len(img_positions) == len(sample["images"]), sample

        new_input_ids = []
        new_targets = []
        st = 0
        for img_idx, img_pos in enumerate(img_positions):
            image_patches, (best_width, best_height) = processor[
                "image"
            ].process_images_with_subpatch(sample["images"][img_idx])
            images.append(image_patches)

            new_input_ids += input_ids[st:img_pos]
            new_targets += targets[st:img_pos]

            new_input_ids += [IMG_START_ID]
            new_targets += [IGNORE_TOKEN_ID]

            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length
            new_targets += [IGNORE_TOKEN_ID] * image_token_length

            new_input_ids += [IMG_END_ID]
            new_targets += [IGNORE_TOKEN_ID]

            if len(image_patches) > 1:
                for i in range(0, best_height, processor["image"].patch_size):
                    new_input_ids += nl_tokens
                    new_targets += [IGNORE_TOKEN_ID] * len(nl_tokens)

                    for j in range(0, best_width, processor["image"].patch_size):
                        new_input_ids += [PATCH_START_ID]
                        new_targets += [IGNORE_TOKEN_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(
                                len(new_input_ids), len(new_input_ids) + image_token_length
                            )
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length
                        new_targets += [IGNORE_TOKEN_ID] * image_token_length

                        new_input_ids += [PATCH_END_ID]
                        new_targets += [IGNORE_TOKEN_ID]

            st = img_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    # ----------------------------------------------------------------
    # video
    if has_video(sample):
        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        assert len(vid_positions) == len(sample["videos"]), sample

        new_input_ids = []
        new_targets = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            video_frames, _ = processor["image"].process_video(
                sample["videos"][vid_idx], max_num_frame, max_fps
            )

            new_input_ids += input_ids[st:vid_pos]
            new_targets += targets[st:vid_pos]

            images.append(video_frames)

            for _ in video_frames:
                new_input_ids += [VID_START_ID]
                new_targets += [IGNORE_TOKEN_ID]

                image_indice_b = torch.zeros(
                    1, image_token_length, dtype=torch.int64
                )  # This will change in collate_fn
                image_indice_s = (
                    torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                    .unsqueeze(0)
                    .repeat(1, 1)
                )
                image_indice_b_s = torch.stack(
                    [image_indice_b, image_indice_s], dim=0
                )  # 2, num_image, image_length
                image_indices.append(image_indice_b_s)

                new_input_ids += [VID_CONTEXT_ID] * image_token_length
                new_targets += [IGNORE_TOKEN_ID] * image_token_length

                new_input_ids += [VID_END_ID]
                new_targets += [IGNORE_TOKEN_ID]

            st = vid_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    # ----------------------------------------------------------------
    # audio
    if has_audio(sample) and processor["audio"].is_contiguous:
        aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]
        # assert len(aud_positions) == len(sample["audios"]), sample
        assert len(aud_positions) == len(unused_audio_idxs), sample

        new_input_ids = []
        new_targets = []
        st = 0
        for aud_idx, aud_pos in enumerate(aud_positions):
            aud_idx = unused_audio_idxs[aud_idx]
            audio = processor["audio"].process_audios(sample["audios"][aud_idx], is_contiguous=True)
            audios.append(audio)
            audio_token_length = audio.size(0) + 4
            # audio_token_length = audio.size(0)

            new_input_ids += input_ids[st:aud_pos]
            new_targets += targets[st:aud_pos]

            new_input_ids += [AUD_START_ID]
            new_targets += [IGNORE_TOKEN_ID]

            audio_indice_b = torch.zeros(
                1, audio_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            audio_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + audio_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            audio_indice_b_s = torch.stack(
                [audio_indice_b, audio_indice_s], dim=0
            )  # 2, num_image, image_length
            audio_indices.append(audio_indice_b_s)

            new_input_ids += [AUD_CONTEXT_ID] * audio_token_length
            new_targets += [IGNORE_TOKEN_ID] * audio_token_length

            new_input_ids += [AUD_END_ID]
            new_targets += [IGNORE_TOKEN_ID]

            st = aud_pos + 1

        new_input_ids += input_ids[st:]
        new_targets += targets[st:]

        input_ids = new_input_ids
        targets = new_targets

    if len(images) > 0:
        images = torch.cat(images, dim=0)

    if len(image_indices) > 0:
        image_indices = torch.cat(image_indices, dim=1)

    # if len(audios) > 0:
    #     audios = torch.cat(audios, dim=0)

    # if len(audio_indices) > 0:
    #     audio_indices = torch.cat(audio_indices, dim=1)

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
        audios=audios,
        audio_indices=audio_indices,
    )


def has_video(sample):
    # video
    if (
        "videos" in sample
        and isinstance(sample["videos"], list)
        and None not in sample["videos"]
        and len(sample["videos"])
    ):
        return True
    return False


def has_image(sample):
    # image
    if (
        "images" in sample
        and isinstance(sample["images"], list)
        and None not in sample["images"]
        and len(sample["images"])
    ):
        return True
    return False


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


def text_audio_interval_old(input_ids, AUD_START_ID, AUD_END_ID, text_audio_interval_ratio):

    if text_audio_interval_ratio is not None:
        text_num, audio_num = text_audio_interval_ratio

    else:
        text_num = 13
        audio_num = 26
        text_num = 4
        audio_num = 10

    # exclude AUD_START and AUD_END
    audio_num = audio_num - 2

    st = [i for i, x in enumerate(input_ids) if x == AUD_START_ID]
    ed = [i for i, x in enumerate(input_ids) if x == AUD_END_ID]

    # only text
    if len(st) == 0 and len(ed) == 0:
        return input_ids

    assert len(st) == 1
    assert len(ed) == 1

    st = st[0]
    ed = ed[0]

    assert st < ed

    # only audio
    if st == 0 and ed == len(input_ids) - 1:
        return input_ids

    audio_tokens = input_ids[st + 1 : ed]
    text_tokens = input_ids[:st] + input_ids[ed + 1 :]

    if False:
        audio_tokens_chunks = [
            audio_tokens[i : i + audio_num] for i in range(0, len(audio_tokens), audio_num)
        ]
        text_tokens_chunks = [
            text_tokens[i : i + text_num] for i in range(0, len(text_tokens), text_num)
        ]

    if False:
        # [0 1] [2 3 4 5 6 audio_num-1] ...
        audio_tokens_chunks = [audio_tokens[:2], audio_tokens[2:audio_num]] + [
            audio_tokens[i : i + audio_num] for i in range(audio_num, len(audio_tokens), audio_num)
        ]
        # [0] [1 2 text_num-1] ...
        text_tokens_chunks = [text_tokens[:1], text_tokens[1:text_num]] + [
            text_tokens[i : i + text_num] for i in range(text_num, len(text_tokens), text_num)
        ]

    if True:
        # [0 1 2 3 4 5 6 audio_num] [] ...
        audio_tokens_chunks = [audio_tokens[:audio_num]] + [
            audio_tokens[i : i + audio_num] for i in range(audio_num, len(audio_tokens), audio_num)
        ]
        # [0] [] ...
        text_tokens_chunks = [text_tokens[:1]] + [
            text_tokens[i : i + text_num] for i in range(1, len(text_tokens), text_num)
        ]

    chunk_num = min(len(audio_tokens_chunks), len(text_tokens_chunks))
    audio_tokens_chunks = audio_tokens_chunks[: chunk_num - 1] + [
        sum(audio_tokens_chunks[chunk_num - 1 :], [])
    ]
    text_tokens_chunks = text_tokens_chunks[: chunk_num - 1] + [
        sum(text_tokens_chunks[chunk_num - 1 :], [])
    ]

    interval_input_ids = []
    for text_tokens, audio_tokens in zip(text_tokens_chunks, audio_tokens_chunks):
        interval_input_ids += text_tokens + [AUD_START_ID] + audio_tokens + [AUD_END_ID]
        # interval_input_ids += text_tokens + audio_tokens

    return interval_input_ids


def text_audio_interval(input_ids, AUD_START_ID, AUD_END_ID, text_audio_interval_ratio):

    if text_audio_interval_ratio is None:
        #                            T   A
        text_audio_interval_ratio = [13, 26]
        #                            T  A  T  A  T  A
        text_audio_interval_ratio = [1, 4, 3, 8, 4, 10]
        #                            T  A   T  A
        text_audio_interval_ratio = [1, 10, 4, 10]

    text_nums = text_audio_interval_ratio[::2]
    audio_nums = text_audio_interval_ratio[1::2]

    # exclude AUD_START and AUD_END
    audio_nums = [x - 2 for x in audio_nums]

    st = [i for i, x in enumerate(input_ids) if x == AUD_START_ID]
    ed = [i for i, x in enumerate(input_ids) if x == AUD_END_ID]

    # only text
    if len(st) == 0 and len(ed) == 0:
        return input_ids

    assert len(st) == 1
    assert len(ed) == 1

    st = st[0]
    ed = ed[0]

    assert st < ed

    # only audio
    if st == 0 and ed == len(input_ids) - 1:
        return input_ids

    audio_tokens = input_ids[st + 1 : ed]
    text_tokens = input_ids[:st] + input_ids[ed + 1 :]

    audio_tokens_chunks = []
    while len(audio_tokens) > 0:
        if len(audio_nums) > 1:
            audio_num = audio_nums.pop(0)
        else:
            audio_num = audio_nums[0]

        audio_tokens_chunks.append(audio_tokens[:audio_num])
        audio_tokens = audio_tokens[audio_num:]

    text_tokens_chunks = []
    while len(text_tokens) > 0:
        if len(text_nums) > 1:
            text_num = text_nums.pop(0)
        else:
            text_num = text_nums[0]

        text_tokens_chunks.append(text_tokens[:text_num])
        text_tokens = text_tokens[text_num:]

    chunk_num = min(len(audio_tokens_chunks), len(text_tokens_chunks))
    audio_tokens_chunks = audio_tokens_chunks[: chunk_num - 1] + [
        sum(audio_tokens_chunks[chunk_num - 1 :], [])
    ]
    text_tokens_chunks = text_tokens_chunks[: chunk_num - 1] + [
        sum(text_tokens_chunks[chunk_num - 1 :], [])
    ]

    interval_input_ids = []
    for text_tokens, audio_tokens in zip(text_tokens_chunks, audio_tokens_chunks):
        interval_input_ids += text_tokens + [AUD_START_ID] + audio_tokens + [AUD_END_ID]
        # interval_input_ids += text_tokens + audio_tokens

    return interval_input_ids
