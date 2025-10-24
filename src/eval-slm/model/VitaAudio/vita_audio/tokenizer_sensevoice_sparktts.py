import logging
import os
import uuid

import torch

import torchaudio

from .constants import (
    AUD_CONTEXT_TOKEN,
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


def update_tokenizer_for_sensevoice_sparktts(tokenizer):
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
        AUD_CONTEXT_TOKEN,
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

    token_list = [f"<|audio_{i}|>" for i in range(8192)]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=False)

    # logger.info(f"tokenizer {tokenizer}")
    return tokenizer


class SenseVoiceSparkTTSTokenizer:
    def __init__(self, model_name_or_path, rank=None):
        self.model_name_or_path = model_name_or_path

        if rank is None and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            self.rank = rank % 8
        else:
            self.rank = rank
        logger.info(f"{self.rank=}")

        self.sampling_rate = 16000

        self.is_discrete = True
        self.is_contiguous = True

        #                            T  A   T  A
        text_audio_interval_ratio = [1, 10, 1, 10]

        self.text_audio_interval_ratio = text_audio_interval_ratio

    def load_model(self):
        if hasattr(self, "model"):
            return

        if self.rank is not None:
            self.device = f"cuda:{self.rank}"
            torch.cuda.set_device(self.rank)
        else:
            self.device = "cpu"
        logger.info(f"{self.device=}")

        logger.info("Loading SenseVoiceSmall")
        from funasr.models.sense_voice.model import SenseVoiceSmall

        model_dir = "/data/models/FunAudioLLM/SenseVoiceSmall/"
        _, self.kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=self.device)
        logger.info("Loading SenseVoiceSmall Done")

        logger.info("Loading BiCodecTokenizer")
        from sparktts.models.audio_tokenizer import BiCodecTokenizer

        model_dir = "/data/models/SparkAudio/Spark-TTS-0.5B/"
        # import time
        # import random
        # time.sleep(self.rank * 2 + random.randint(3, 9))
        self.model = BiCodecTokenizer(model_dir, device=self.device)
        logger.info("Loading BiCodecTokenizer Done")

    def encode(self, audio_path, is_discrete=False, is_contiguous=True, **kwargs):
        if not hasattr(self, "model"):
            self.load_model()

        assert not (is_discrete and is_contiguous)
        assert is_discrete or is_contiguous

        if is_discrete:
            global_token_ids, semantic_token_ids = self.model.tokenize(audio_path)

            semantic_token_ids = semantic_token_ids[0].cpu().tolist()
            return semantic_token_ids

        if is_contiguous:
            from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

            audio, sampling_rate = torchaudio.load(audio_path)
            audio = audio.mean(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=self.sampling_rate
            )
            audio = resampler(audio[None, :])[0, :]
            # audio = audio.to(self.device)

            frontend = self.kwargs["frontend"]

            speech, speech_lengths = extract_fbank(audio, data_type="sound", frontend=frontend)

            speech = speech[0]
            # print(f"{speech_lengths=}")
            # print(f"{speech.size()=}")

            return speech

    def decode(self, prompt_speech_token, source_speech_16k=None):
        if not hasattr(self, "model"):
            self.load_model()

        semantic_token_ids = torch.tensor(prompt_speech_token, dtype=torch.long).unsqueeze(0)
        # print(f"{semantic_token_ids=}")

        if source_speech_16k is None:
            global_token_ids = torch.zeros((1, 1, 32), dtype=torch.long)
        else:
            global_token_ids, _ = self.model.tokenize(source_speech_16k)
        # print(f"{source_speech_16k=}")
        print(f"{global_token_ids=}")

        audio = self.model.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            semantic_token_ids.to(self.device),
        )

        print(f"{audio=}")
        # audio = torch.tensor(audio).unsqueeze(0)
        audio = torch.tensor(audio)

        return audio

    def apply_to_role(self, role, **kwargs):
        is_discrete = kwargs.get("is_discrete", False)
        if is_discrete and role in ["assistant", "gpt"]:
            return True

        is_contiguous = kwargs.get("is_contiguous", False)
        if is_contiguous and role in ["user", "human"]:
            return True

        return False
