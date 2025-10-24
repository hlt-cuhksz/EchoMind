import json
import math
import os

import numpy as np
import torch

import natsort
from vita_audio.tokenizer import get_audio_tokenizer


class AudioProcessor:
    def __init__(
        self,
        audio_tokenizer_path=None,
        audio_tokenizer_type=None,
        text_audio_interval_ratio=None,
    ):

        self.audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
        )

        self.audio_tokenizer_type = audio_tokenizer_type

        self.text_audio_interval_ratio = text_audio_interval_ratio

        # self.load_model()

    def load_model(self):
        if self.audio_tokenizer is not None:
            self.audio_tokenizer.load_model()

    def process_audios(self, audio_path, is_discrete=False, is_contiguous=False, **kwargs):

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete:
            audio_tokenizer_type = self.audio_tokenizer_type.split("_")[-1]
            cache_path = os.path.splitext(audio_path)[0] + f"_{audio_tokenizer_type}.json"
            try:
                if os.path.isfile(cache_path):
                    with open(cache_path, "r") as f:
                        audio_data = json.load(f)
                    return audio_data
            except Exception as e:
                pass

        audio_data = self.audio_tokenizer.encode(
            audio_path, is_discrete=is_discrete, is_contiguous=is_contiguous, **kwargs
        )
        # print(f"{len(audio_data)=}")

        if is_discrete:
            try:
                if isinstance(audio_data, list):
                    with open(cache_path, "w") as f:
                        json.dump(audio_data, f)
            except Exception as e:
                pass

        return audio_data

    @property
    def is_discrete(self):
        return self.audio_tokenizer.is_discrete

    @property
    def is_contiguous(self):
        return self.audio_tokenizer.is_contiguous

    def apply_to_role(self, role, **kwargs):
        return self.audio_tokenizer.apply_to_role(role, **kwargs)

    def text_audio_interval(self, content_input_id, AUD_START_ID, AUD_END_ID):

        return text_audio_interval(
            content_input_id,
            AUD_START_ID,
            AUD_END_ID,
            self.text_audio_interval_ratio,
        )


def add_audio_input_contiguous(input_ids, audio_paths, tokenizer, audio_tokenizer):

    from ...constants import (
        AUD_START_TOKEN,
        AUD_END_TOKEN,
        AUD_TAG_TOKEN,
        AUD_CONTEXT_TOKEN,
    )

    AUD_CONTEXT_ID = tokenizer(AUD_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    AUD_TAG_ID = tokenizer(AUD_TAG_TOKEN, add_special_tokens=False).input_ids
    AUD_START_ID = tokenizer(AUD_START_TOKEN, add_special_tokens=False).input_ids
    AUD_END_ID = tokenizer(AUD_END_TOKEN, add_special_tokens=False).input_ids

    AUD_CONTEXT_ID = AUD_CONTEXT_ID[0]
    AUD_TAG_ID = AUD_TAG_ID[0]
    AUD_START_ID = AUD_START_ID[0]
    AUD_END_ID = AUD_END_ID[0]

    aud_positions = [i for i, x in enumerate(input_ids) if x == AUD_TAG_ID]

    audios = []
    audio_indices = []
    new_input_ids = []
    st = 0
    for aud_idx, aud_pos in enumerate(aud_positions):
        audio = audio_tokenizer.encode(audio_paths[aud_idx], is_contiguous=True)
        audios.append(audio)
        audio_token_length = audio.size(0) + 4

        new_input_ids += input_ids[st:aud_pos]

        new_input_ids += [AUD_START_ID]

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

        new_input_ids += [AUD_END_ID]

        st = aud_pos + 1

    new_input_ids += input_ids[st:]
    inputs_ids = new_input_ids

    return inputs_ids, audios, audio_indices


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
