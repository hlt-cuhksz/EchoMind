"""
Feature extractor class for Whisper
"""
import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np

from transformers import WhisperFeatureExtractor as HFWhisperFeatureExtractor
from transformers import is_torch_available
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, logging


logger = logging.get_logger(__name__)


class WhisperFeatureExtractor(HFWhisperFeatureExtractor):
    r"""
    The original WhisperFeatureExtractor only support audio whose duration is less than 30s.
    For example, 15s -> 30s; 45s -> 30s.
    Here, we will pad the audio to 30n s.
    For example, 15s -> 30s; 45s -> 60s.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        hop_length=160,
        chunk_length=30,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=False,  # pad inputs to max length with silence token (zero) and no attention mask
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            hop_length=hop_length,
            chunk_length=chunk_length,
            n_fft=n_fft,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
    
    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        truncation: bool = True,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        padding: Optional[str] = "max_length",
        max_length: Optional[int] = None,
        sampling_rate: Optional[int] = None,
        do_normalize: Optional[bool] = None,
        device: Optional[str] = "cpu",
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32).T for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        batched_speech = BatchFeature({"input_features": raw_speech})

        # pad to self.n_samples*n
        # for example: 240000 -> 480000; 720000 -> 960000
        length = max(s.shape[0] for s in raw_speech)
        length = ((length - 1) // self.n_samples + 1) * self.n_samples

        # convert into correct format for padding

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=max_length if max_length else length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=return_attention_mask or do_normalize,
        )

        # zero-mean and unit-variance normalization
        if do_normalize:
            padded_inputs["input_features"] = self.zero_mean_unit_var_norm(
                padded_inputs["input_features"],
                attention_mask=padded_inputs["attention_mask"],
                padding_value=self.padding_value,
            )
            padded_inputs["input_features"] = np.stack(padded_inputs["input_features"], axis=0)

        # make sure list is in array format
        input_features = padded_inputs.get("input_features").transpose(2, 0, 1)

        extract_fbank_features = (
            self._torch_extract_fbank_features if is_torch_available() else self._np_extract_fbank_features
        )
        input_features = extract_fbank_features(input_features[0], device)

        # input_features = [self._np_extract_fbank_features(waveform) for waveform in input_features[0]]

        if isinstance(input_features[0], List):
            padded_inputs["input_features"] = [np.asarray(feature, dtype=np.float32) for feature in input_features]
        else:
            padded_inputs["input_features"] = input_features

        if return_attention_mask:
            # rescale from sample (48000) to feature (3000)
            padded_inputs["attention_mask"] = padded_inputs["attention_mask"][:, :: self.hop_length]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


FEATURE_EXTRACTION_MAPPING = {
    "whisper": WhisperFeatureExtractor,
    "qwen2_audio_encoder": WhisperFeatureExtractor,
    "qwen2_audio": WhisperFeatureExtractor
}