import logging
import os
import uuid

import torch

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


def update_tokenizer_for_cosyvoice2(tokenizer):
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

    token_list = [f"<|audio_{i}|>" for i in range(6561)]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=False)

    # logger.info(f"tokenizer {tokenizer}")
    return tokenizer


class CosyVoice2Tokenizer:
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
        if hasattr(self, "cosyvoice"):
            return

        logger.info("Loading CosyVoice2Tokenizer")
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
        from cosyvoice.utils.file_utils import load_wav

        if self.rank is not None:
            torch.cuda.set_device(self.rank)
        else:
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(f"{self.rank}")

        self.cosyvoice = CosyVoice2(
            self.model_name_or_path, load_jit=False, load_trt=False, fp16=True
        )
        del self.cosyvoice.model.llm

        self.load_wav = load_wav

    def encode(self, audio_path, **kwargs):
        if not hasattr(self, "cosyvoice"):
            self.load_model()

        speech_16k = self.load_wav(audio_path, 16000)

        try:
            speech_token, speech_token_len = self.cosyvoice.frontend._extract_speech_token(
                speech_16k
            )
            speech_token = speech_token[0].cpu().tolist()
        except Exception as error:
            # logger.info("error", error)
            speech_token = []
        # logger.info(f"speech_token {speech_token}")

        return speech_token

    def decode(self, prompt_speech_token, source_speech_16k=None):
        if not hasattr(self, "cosyvoice"):
            self.load_model()

        prompt_speech_token = torch.tensor(prompt_speech_token).unsqueeze(0)

        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)

        prompt_speech_feat = torch.zeros(1, 0, 80)

        if source_speech_16k is None:
            flow_embedding = torch.zeros(1, 192)
        else:
            flow_embedding = self.cosyvoice.frontend._extract_spk_embedding(source_speech_16k)

        this_uuid = str(uuid.uuid1())
        this_uuid = "abc"
        self.cosyvoice.model.hift_cache_dict[this_uuid] = None

        token_offset = 0

        tts_speech = self.cosyvoice.model.token2wav(
            token=prompt_speech_token,
            prompt_token=flow_prompt_speech_token,
            prompt_feat=prompt_speech_feat,
            embedding=flow_embedding,
            uuid=this_uuid,
            token_offset=token_offset,
            finalize=True,
        )

        tts_speech = tts_speech.squeeze().cpu()

        return tts_speech

    def apply_to_role(self, role, **kwargs):
        is_discrete = kwargs.get("is_discrete", False)
        if is_discrete:
            return True

        is_contiguous = kwargs.get("is_contiguous", False)
        if is_contiguous:
            return False

        return True
