import logging
import os
import uuid

import torch

import torchaudio

from .constants import (
    AUD_END_TOKEN,
    AUD_START_TOKEN,
    AUD_TAG_TOKEN,
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    IMG_TAG_TOKEN,
    PATCH_CONTEXT_TOKEN,
    PATCH_END_TOKEN,
    PATCH_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    VID_CONTEXT_TOKEN,
    VID_END_TOKEN,
    VID_START_TOKEN,
    VID_TAG_TOKEN,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def update_tokenizer_for_snac(tokenizer):
    token_list = [
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
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        IMG_TAG_TOKEN,
        VID_TAG_TOKEN,
        AUD_TAG_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)

    token_list = [f"<|audio_{i}|>" for i in range(4 * 4096)]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=False)

    # logger.info(f"tokenizer {tokenizer}")
    return tokenizer


class SNACTokenizer:
    def __init__(self, model_name_or_path, rank=None):
        self.model_name_or_path = model_name_or_path

        if rank is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            self.rank = rank % 8
        else:
            self.rank = rank
        logger.info(f"{self.rank=}")

        self.is_discrete = True
        self.is_contiguous = False

        #                            T   A
        text_audio_interval_ratio = [13, 26]

        self.text_audio_interval_ratio = text_audio_interval_ratio

    def load_model(self):
        if hasattr(self, "model"):
            return

        logger.info("Loading SNACTokenizer")
        from snac import SNAC

        self.device = f"cuda:{self.rank}"
        torch.cuda.set_device(self.rank)

        self.model = SNAC.from_pretrained(self.model_name_or_path).eval().to(self.device)

    def encode(self, audio_path, **kwargs):
        if not hasattr(self, "model"):
            self.load_model()

        audio, sampling_rate = torchaudio.load(audio_path)
        audio = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, new_freq=self.model.sampling_rate
        )(audio)
        audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        with torch.inference_mode():
            codes = self.model.encode(audio)

        codes = shift_code(codes, self.model.codebook_size, self.model.vq_strides)
        audio_tokens = codes.cpu().tolist()

        return audio_tokens

    def decode(self, audio_tokens, **kwargs):
        if not hasattr(self, "model"):
            self.load_model()

        while len(audio_tokens) % sum(self.model.vq_strides):
            audio_tokens += [
                audio_tokens[-1] + 4096,
            ]
        codes = torch.tensor(audio_tokens, device=self.device)
        codes = inverse_shift_code(codes, self.model.codebook_size, self.model.vq_strides)
        codes = [torch.clamp(x, min=0, max=self.model.codebook_size - 1) for x in codes]
        # logger.info(f"codes {codes} {[x.size() for x in codes]}")

        with torch.inference_mode():
            audio_hat = self.model.decode(codes)

        # logger.info(f"audio_hat {audio_hat.size()}")
        audio_hat = audio_hat.squeeze(0).squeeze(0).cpu()
        return audio_hat

    def apply_to_role(self, role, **kwargs):
        is_discrete = kwargs.get("is_discrete", False)
        if is_discrete:
            return True

        is_contiguous = kwargs.get("is_contiguous", False)
        if is_contiguous:
            return False

        return True


def shift_code(codes, codebook_size, vq_strides):
    # codes: [torch.Size([1, 43]), torch.Size([1, 86]), torch.Size([1, 172])]

    # 3 * 4096 new vocabularies
    # codes = torch.cat([x.reshape(1, -1, vq_strides[-i-1]) + i * codebook_size for i, x in enumerate(codes)], dim=-1).reshape(-1)

    # 7 * 4096 new vocabularies
    codes = [x.reshape(1, -1, s) for s, x in zip(vq_strides[::-1], codes)]
    codes = torch.cat(
        [
            x + i * codebook_size
            for i, x in enumerate(torch.cat(codes, dim=-1).chunk(sum(vq_strides), dim=-1))
        ],
        dim=-1,
    ).reshape(-1)

    return codes


def inverse_shift_code(codes, codebook_size, vq_strides):
    # codes: torch.Size([301])

    # 3 * 4096 new vocabularies
    # codes = [x.reshape(1, -1) - i * codebook_size for i, x in enumerate(codes.reshape(1, -1, sum(vq_strides)).split(vq_strides[::-1], dim=-1))]

    # 7 * 4096 new vocabularies
    codes = torch.cat(
        [
            x - i * codebook_size
            for i, x in enumerate(
                codes.reshape(1, -1, sum(vq_strides)).chunk(sum(vq_strides), dim=-1)
            )
        ],
        dim=-1,
    ).split(vq_strides[::-1], dim=-1)
    codes = [x.reshape(1, -1) for x in codes]

    return codes
