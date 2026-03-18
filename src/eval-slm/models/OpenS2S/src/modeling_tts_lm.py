from typing import Optional, Tuple, Union
from dataclasses import dataclass
import os
import glob

import torch
import safetensors
import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack

from transformers import Qwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM
from transformers import Qwen3Config
from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

from src.constants import IGNORE_INDEX


class Qwen2ForCausalLM(HFQwen2ForCausalLM):
    """
    overwrite forward to support spk_emb
    """

    def __init__(self, config):
        super().__init__(config)

        self.spk2tts = nn.Linear(512, self.config.hidden_size, bias=False)
        self.spk_ln = nn.LayerNorm(self.config.hidden_size, 1e-5, True)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        spk_embs: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        if inputs_embeds is not None:
            return super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs
            )
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            spk_embs = self.spk_ln(self.spk2tts(spk_embs))
            inputs_embeds = torch.cat(
                [spk_embs, inputs_embeds],
                dim=1
            )
            attention_mask = torch.cat(
                [
                    torch.ones(attention_mask.shape[0], 1).to(attention_mask.dtype).to(attention_mask.device),
                    attention_mask
                ],
                dim=1
            )
            labels = torch.cat(
                [
                    torch.zeros(labels.shape[0], 1).fill_(IGNORE_INDEX).to(labels.dtype).to(labels.device),
                    labels
                ],
                dim=1
            )

            return super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs
            )

class Qwen3ForCausalLM(HFQwen3ForCausalLM):
    """
    overwrite forward to support spk_emb
    """

    def __init__(self, config):
        super().__init__(config)

        self.spk2tts = nn.Linear(512, self.config.hidden_size, bias=False)
        self.spk_ln = nn.LayerNorm(self.config.hidden_size, 1e-5, True)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        spk_embs: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ):
        if inputs_embeds is not None:
            return super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs
            )
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            # spk_embs = self.spk_ln(self.spk2tts(spk_embs))
            # inputs_embeds = torch.cat(
            #     [spk_embs, inputs_embeds],
            #     dim=1
            # )
            # attention_mask = torch.cat(
            #     [
            #         torch.ones(attention_mask.shape[0], 1).to(attention_mask.dtype).to(attention_mask.device),
            #         attention_mask
            #     ],
            #     dim=1
            # )
            # labels = torch.cat(
            #     [
            #         torch.zeros(labels.shape[0], 1).fill_(IGNORE_INDEX).to(labels.dtype).to(labels.device),
            #         labels
            #     ],
            #     dim=1
            # )

            output =  super().forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs
            )
            return output

TTS_LM_MAPPING = {
    "qwen2": Qwen2ForCausalLM,
    "qwen3": Qwen3ForCausalLM
}