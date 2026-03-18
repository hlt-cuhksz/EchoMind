from typing import Optional, Tuple
from dataclasses import dataclass
import os
import glob

import torch
import safetensors
import torch
from transformers.utils import ModelOutput


from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder as HFWhisperEncoder
from transformers import Qwen2AudioEncoderConfig
from transformers import Qwen2AudioEncoder as HFQwen2AudioEncoder


from .utils import length_to_attention_mask

@dataclass
class AudioEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    output_lengths: Optional[torch.LongTensor] = None


class WhisperEncoder(HFWhisperEncoder):
    """
    overwrite forward to support long audio
    overwrite from_pretrained to support split encoder parameters from pretrained WhisperModel
    """

    def from_pretrained(model_path):
        config = WhisperConfig.from_pretrained(model_path)

        model = WhisperEncoder(config)
        old_state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        state_dict = {}
        for para_name in old_state_dict.keys():
            if "model.encoder." in para_name:
                new_name = para_name.replace("model.encoder.", "")
                state_dict[new_name] = old_state_dict[para_name]
        model.load_state_dict(state_dict)

        return model


    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        bz, hidden_dim, _ = input_features.size()
        input_features = input_features.transpose(1,2) # B x C x T -> B x T x C
        input_n_samples = self.max_source_positions * 2 # 3000

        input_lengths = attention_mask.sum(-1)
        segments = torch.ceil(input_lengths / input_n_samples).to(dtype=torch.long)
        input_features = input_features.contiguous().view(-1, input_n_samples, hidden_dim) # M x 3000 x C
        attention_mask = attention_mask.contiguous().view(-1, input_n_samples) # M x 3000
        ### filter empty input
        ### for example, when a1(15s) and a2(45s) in one batch
        ### the audios will be padded into 60s and then segmented into 30s like [a11, a12, a21, a22]
        ### we will skip the computation for a12
        select_index = length_to_attention_mask(segments).to(torch.bool).view(-1)
        input_features = input_features[select_index]
        attention_mask = attention_mask[select_index]
        input_features = input_features.transpose(1,2) # M x 3000 x C -> M x C x 3000

        output = super().forward(
            input_features,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        last_hidden_state = output.last_hidden_state # M x 1500 x C
        input_lengths = attention_mask.sum(-1) # M
        output_n_samples = last_hidden_state.size(1)
        ### concate the faetures of the same example
        output_last_hidden_state = last_hidden_state.new_zeros(
            bz, output_n_samples * segments.max().item(), last_hidden_state.size(2)
        )
        output_lengths = input_lengths.new_zeros(bz)
        idx = 0
        for i, l in enumerate(segments):
            output_last_hidden_state[i,:output_n_samples*l,:] = \
                last_hidden_state[idx:idx+l].contiguous().view(-1, last_hidden_state.size(2))
            output_lengths[i] = self._get_feat_extract_output_lengths(input_lengths[idx:idx+l]).sum()
            idx += l

        max_length = output_lengths.max()
        output_last_hidden_state = output_last_hidden_state[:,:max_length,:]

        return AudioEncoderOutput(
            last_hidden_state=output_last_hidden_state,
            hidden_states=None,
            attentions=None,
            output_lengths=output_lengths
        )


class Qwen2AudioEncoder(HFQwen2AudioEncoder):
    """
        overwrite forward to support long audio
        overwrite from_pretrained to support split encoder parameters from pretrained Qwen2Audio
        """

    def from_pretrained(model_path):
        config = Qwen2AudioEncoderConfig.from_pretrained(model_path)

        model = Qwen2AudioEncoder(config)
        state_dict = {}
        for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "audio_tower" in key:
                        new_key = key.replace("audio_tower.", "")
                        state_dict[new_key] = f.get_tensor(key)
        model.load_state_dict(state_dict)

        return model


    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        bz, hidden_dim, _ = input_features.size()
        input_features = input_features.transpose(1,2) # B x C x T -> B x T x C
        input_n_samples = self.max_source_positions * 2 # 3000

        input_lengths = attention_mask.sum(-1)
        segments = torch.ceil(input_lengths / input_n_samples).to(dtype=torch.long)
        input_features = input_features.contiguous().view(-1, input_n_samples, hidden_dim) # M x 3000 x C
        attention_mask = attention_mask.contiguous().view(-1, input_n_samples) # M x 3000
        ### filter empty input
        ### for example, when a1(15s) and a2(45s) in one batch
        ### the audios will be padded into 60s and then segmented into 30s like [a11, a12, a21, a22]
        ### we will skip the computation for a12
        select_index = length_to_attention_mask(segments).to(torch.bool).view(-1)
        input_features = input_features[select_index]
        attention_mask = attention_mask[select_index]
        input_features = input_features.transpose(1,2) # M x 3000 x C -> M x C x 3000

        output = super().forward(
            input_features,
            None,  ## qwen2_audio donot support attention_mask
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        last_hidden_state = output.last_hidden_state # M x 1500 x C
        input_lengths = attention_mask.sum(-1) # M
        output_n_samples = last_hidden_state.size(1)
        ### concate the faetures of the same example
        output_last_hidden_state = last_hidden_state.new_zeros(
            bz, output_n_samples * segments.max().item(), last_hidden_state.size(2)
        )
        output_lengths = input_lengths.new_zeros(bz)
        idx = 0
        for i, l in enumerate(segments):
            output_last_hidden_state[i,:output_n_samples*l,:] = \
                last_hidden_state[idx:idx+l].contiguous().view(-1, last_hidden_state.size(2))
            output_lengths[i] = self._get_feat_extract_output_lengths(input_lengths[idx:idx+l]).sum()
            idx += l

        max_length = output_lengths.max()
        output_last_hidden_state = output_last_hidden_state[:,:max_length,:]

        return AudioEncoderOutput(
            last_hidden_state=output_last_hidden_state,
            hidden_states=None,
            attentions=None,
            output_lengths=output_lengths
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        return super()._get_feat_extract_output_lengths(input_lengths)[1] # after avg pooler


AUDIO_ENCODER_MAPPING = {
    "whisper": WhisperEncoder,
    "qwen2_audio_encoder": Qwen2AudioEncoder,
    "qwen2_audio": Qwen2AudioEncoder
}