import math
from typing import List, Optional, Tuple, Union
import re
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np

import logging
from transformers import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GenerationConfig
import torch.nn.functional as F
from peft import LoraConfig, LoraModel

from .modeling_adapter import Subsampler
from .utils import length_to_attention_mask
from .configuration_omnispeech import OmniSpeechConfig
from .modeling_audio_encoder import AUDIO_ENCODER_MAPPING
from .modeling_tts_lm import TTS_LM_MAPPING
from .constants import IGNORE_INDEX, AUDIO_TOKEN_INDEX

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class OmniSpeechModel(PreTrainedModel):
    config_class = OmniSpeechConfig
    base_model_prefix = "omnispeech"

    def __init__(self, config: OmniSpeechConfig):
        super().__init__(config)
        self.audio_encoder_config = config.audio_encoder_config
        self.llm_config = config.llm_config
        self.tts_lm_config = config.tts_lm_config
        self.text_span = int(config.interleave_strategy.split(":")[0])
        self.speech_span = int(config.interleave_strategy.split(":")[1])
        self.llm_weight = 1.0
        self.tts_weight = 1.0

        self.audio_encoder_model = AUDIO_ENCODER_MAPPING[self.audio_encoder_config.model_type](self.audio_encoder_config)
        # self.llm_model = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[self.llm_config.model_type](self.llm_config)
        self.llm_model = AutoModelForCausalLM.from_config(self.llm_config)
        self.tts_lm_model = TTS_LM_MAPPING[self.tts_lm_config.model_type](self.tts_lm_config)

        if config.lora_config:
            self.lora_config = config.lora_config
            self.llm_model = LoraModel(self.llm_model, self.lora_config, "default")

        self.audio_adapter = Subsampler(self.audio_encoder_config.d_model, config.adapter_inner_dim,
            self.llm_config.hidden_size, config.conv_kernel_sizes)

        self.llm2tts = nn.Linear(self.llm_config.hidden_size, self.tts_lm_config.hidden_size, bias=False)
        self.llm_ln = nn.LayerNorm(self.tts_lm_config.hidden_size, 1e-5, True)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        token_types: torch.LongTensor,
        labels: torch.LongTensor,
        speech_values: torch.FloatTensor,
        speech_mask: torch.LongTensor,
        speech_units: torch.LongTensor,
        speech_units_mask: torch.LongTensor,
        spk_embs: torch.FloatTensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        (
            inputs_embeds,
            attention_mask,
            labels,
            token_types
        ) = self.prepare_inputs_labels_for_llm(
            input_ids,
            attention_mask,
            labels,
            speech_values,
            speech_mask,
            token_types
        )

        llm_output = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=labels,
            output_hidden_states=True
        )

        if speech_units is None:
            batch_size = inputs_embeds.shape[0]
            ### dummy 2 token tts
            dummy_speech_units = inputs_embeds.new_ones((batch_size, 2)).to(torch.long)
            tts_inputs_embeds = self.tts_lm_model.get_input_embeddings()(dummy_speech_units)
            tts_attention_mask = dummy_speech_units.new_ones((batch_size, 2))
            tts_labels = dummy_speech_units.new_ones((batch_size, 2))

            tts_lm_output = self.tts_lm_model(
                inputs_embeds=tts_inputs_embeds,
                attention_mask=tts_attention_mask,
                return_dict=True,
                labels=tts_labels
            )
            llm_output.loss = llm_output.loss + 0 * tts_lm_output.loss
            return llm_output
        else:
            hidden_states = llm_output.hidden_states[-1]
            tts_inputs_embeds, tts_attention_mask, tts_labels = self.prepare_inputs_labels_for_tts_lm(
                hidden_states, speech_units, speech_units_mask, spk_embs, token_types
            )

            tts_lm_output = self.tts_lm_model(
                inputs_embeds=tts_inputs_embeds,
                attention_mask=tts_attention_mask,
                return_dict=True,
                labels=tts_labels
            )
            # return tts_lm_output

            llm_output.loss = self.llm_weight * llm_output.loss + self.tts_weight * tts_lm_output.loss

            return llm_output


    def merge_lora(self):
        if hasattr(self, 'lora_config'):
            self.llm_model = self.llm_model.merge_and_unload()
            self.config.lora_config = None
            del self.lora_config
        else:
            raise ValueError("cannot call merge_lora when no self.lora_config is set")
    
    def add_lora(self, lora_config):
        self.lora_config = lora_config
        self.config.lora_config = lora_config
        self.llm_model = LoraModel(self.llm_model, lora_config, "default")


    def get_speech_features(self, speech_values, speech_attention_mask):
        w2v_args = {
            "input_features": speech_values,
            "attention_mask": speech_attention_mask,
        }
        output = self.audio_encoder_model(**w2v_args)
        speech_embeds = output.last_hidden_state # B x T x C
        attention_mask = length_to_attention_mask(output.output_lengths)

        speech_embeds, speech_atts = self.audio_adapter(speech_embeds, attention_mask)

        return speech_embeds, speech_atts


    def prepare_inputs_labels_for_llm(
        self, input_ids, attention_mask, labels, speech_values, speech_mask, token_types, left_pad=False, inference=False
    ):
        if speech_values is not None:
            speech_features, speech_attention_mask = self.get_speech_features(speech_values, speech_mask)
        else:
            inputs_embeds = self.llm_model.get_input_embeddings()(input_ids)
            if inference:
                return inputs_embeds, attention_mask, labels, token_types
            ### dummy 1s speech
            batch_size = inputs_embeds.shape[0]
            dummy_speech_values = inputs_embeds.new_zeros((batch_size, 128, 3000))
            dummy_speech_mask = inputs_embeds.new_zeros((batch_size, 3000)).to(torch.long)
            dummy_speech_mask[:,:100] = 1
            speech_features, speech_attention_mask = self.get_speech_features(dummy_speech_values, dummy_speech_mask)
            speech_labels = speech_attention_mask.new_zeros((batch_size, speech_attention_mask.shape[1])).fill_(IGNORE_INDEX)
            speech_token_types = speech_attention_mask.new_zeros((batch_size, speech_attention_mask.shape[1]))

            inputs_embeds = torch.cat([inputs_embeds,speech_features], dim=1)
            if attention_mask is None:
                attention_mask = inputs_embeds.new_ones((batch_size, inputs_embeds.shape[1])).to(torch.long)
            attention_mask = torch.cat([attention_mask, speech_attention_mask], dim=1)
            if labels is None:
                labels = inputs_embeds.new_zeros((batch_size, inputs_embeds.shape[1])).to(torch.long).fill_(IGNORE_INDEX)
            labels = torch.cat([labels, speech_labels], dim=1)
            if token_types is None:
                token_types = inputs_embeds.new_zeros((batch_size, inputs_embeds.shape[1])).to(torch.long)
            token_types = torch.cat([token_types, speech_token_types], dim=1)

            return inputs_embeds, attention_mask, labels, token_types
        
        _labels = labels
        _token_types = token_types
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        if token_types is None:
            token_types = torch.full_like(input_ids, 0)

        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        token_types = [cur_token_types[cur_attention_mask] for cur_token_types, cur_attention_mask in zip(token_types, attention_mask)]
        speech_lengths = speech_attention_mask.sum(-1)
        speech_features = [speech_features[i, :speech_lengths[i]] for i in range(len(speech_features))]

        new_inputs_embeds, new_labels, new_token_types = [], [], []
        cur_speech_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_speech = (cur_input_ids == AUDIO_TOKEN_INDEX).sum()
            speech_token_indices = [-1] + torch.where(cur_input_ids == AUDIO_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_nospeech = []
            cur_labels = labels[batch_idx]
            cur_labels_nospeech = []
            cur_token_types = token_types[batch_idx]
            cur_token_types_nospeech = []
            for i in range(len(speech_token_indices) - 1):
                cur_input_ids_nospeech.append(cur_input_ids[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_labels_nospeech.append(cur_labels[speech_token_indices[i]+1:speech_token_indices[i+1]])
                cur_token_types_nospeech.append(cur_token_types[speech_token_indices[i]+1:speech_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_nospeech]
            cur_inputs_embeds = self.llm_model.get_input_embeddings()(torch.cat(cur_input_ids_nospeech))
            cur_inputs_embeds_no_speech = torch.split(cur_inputs_embeds, split_sizes, dim=0)
            cur_new_inputs_embeds = []
            cur_new_labels = []
            cur_new_token_types = []

            for i in range(num_speech + 1):
                cur_new_inputs_embeds.append(cur_inputs_embeds_no_speech[i])
                cur_new_labels.append(cur_labels_nospeech[i])
                cur_new_token_types.append(cur_token_types_nospeech[i])
                if i < num_speech:
                    cur_speech_features = speech_features[cur_speech_idx]
                    cur_speech_idx += 1
                    cur_new_inputs_embeds.append(cur_speech_features)
                    cur_new_labels.append(torch.full((cur_speech_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_token_types.append(torch.full((cur_speech_features.shape[0],), 0, device=cur_token_types.device, dtype=cur_token_types.dtype))
                
            cur_new_inputs_embeds = [x.to(self.device) for x in cur_new_inputs_embeds]
            cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_token_types = torch.cat(cur_new_token_types)

            new_inputs_embeds.append(cur_new_inputs_embeds)
            new_labels.append(cur_new_labels)
            new_token_types.append(cur_new_token_types)
        

        assert cur_speech_idx == len(speech_features)

        # Combine them
        max_len = max(x.shape[0] for x in new_inputs_embeds)
        batch_size = len(new_inputs_embeds)

        new_inputs_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_token_types_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_token_types[0].dtype, device=new_token_types[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)

        for i, (cur_new_embed, cur_new_labels, cur_token_types) in enumerate(zip(new_inputs_embeds, new_labels, new_token_types)):
            cur_len = cur_new_embed.shape[0]
            if left_pad:
                new_inputs_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_token_types_padded[i, -cur_len:] = cur_token_types
                    attention_mask[i, -cur_len:] = True
            else:
                new_inputs_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_token_types_padded[i, :cur_len] = cur_token_types
                    attention_mask[i, :cur_len] = True

        new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        if _token_types is None:
            new_token_types = None
        else:
            new_token_types = new_token_types_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        return new_inputs_embeds, attention_mask, new_labels, new_token_types


    def prepare_inputs_labels_for_tts_lm(
        self, hidden_states, speech_units, speech_units_mask, spk_embs, token_types
    ):
        _hidden_states = self.llm_ln(self.llm2tts(hidden_states))
        _spk_embs = self.tts_lm_model.spk_ln(self.tts_lm_model.spk2tts(spk_embs))

        ### select hidden_states based on token_types
        padded_token_types = F.pad(token_types, (1, 1), "constant", False)  # [B, T+2]
        diff = padded_token_types[:, 1:].int() - padded_token_types[:, :-1].int()  # [B, T+1]
        starts = (diff == 1).nonzero(as_tuple=True)
        ends = (diff == -1).nonzero(as_tuple=True)
        batch_indices, start_indices = starts
        end_indices = ends[1]
        assert len(batch_indices) == len(speech_units)

        max_text_length = max([(end - start) for start, end in zip(start_indices, end_indices)])
        max_units_length = speech_units.shape[1]
        max_inter_length = max_text_length + max_units_length
        bz = speech_units.shape[0]
        hidden_size = self.tts_lm_config.hidden_size

        ### 1 for spk emb
        tts_inputs_embeds = torch.zeros(bz, max_inter_length + 1, hidden_size).to(hidden_states.device).to(hidden_states.dtype)
        tts_attention_mask = torch.zeros(bz, max_inter_length + 1, dtype=torch.long).to(hidden_states.device)
        tts_labels = torch.LongTensor(bz, max_inter_length + 1).fill_(IGNORE_INDEX).to(hidden_states.device)

        ### spk_emb
        tts_inputs_embeds[:,0,:] = _spk_embs.squeeze(1)
        tts_attention_mask[:, 0] = 1

        ### speech units emb
        speech_units_length = (speech_units_mask != 0).sum(-1)
        speech_units_embs = self.tts_lm_model.get_input_embeddings()(speech_units)

        for i in range(bz):
            batch_index, start_index, end_index = batch_indices[i], start_indices[i], end_indices[i]
            text_length = end_index - start_index
            text_emb = _hidden_states[batch_index][start_index: end_index]
            units_length = speech_units_length[i].item()
            units_emb = speech_units_embs[i, :units_length, :]

            ### interleave
            index1 = 0
            index2 = 0
            result_index = 1
            while index1 < text_length or index2 < units_length:
                end_index1 = min(index1 + self.text_span, text_length)
                rows_to_copy1 = end_index1 - index1
                if rows_to_copy1 > 0:
                    tts_inputs_embeds[i, result_index:result_index + rows_to_copy1, :] = text_emb[index1:end_index1]
                    tts_attention_mask[i, result_index:result_index + rows_to_copy1] = 1
                    result_index += rows_to_copy1
                index1 = end_index1

                end_index2 = min(index2 + self.speech_span, units_length)
                rows_to_copy2 = end_index2 - index2
                if rows_to_copy2 > 0:
                    tts_inputs_embeds[i, result_index:result_index + rows_to_copy2, :] = units_emb[index2:end_index2]
                    tts_attention_mask[i, result_index:result_index + rows_to_copy2] = 1
                    tts_labels[i, result_index:result_index + rows_to_copy2] = speech_units[i, index2:end_index2]
                    result_index += rows_to_copy2
                index2 = end_index2

        return tts_inputs_embeds, tts_attention_mask, tts_labels


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        speech_values: torch.FloatTensor,
        speech_mask: torch.LongTensor,
        spk_emb: torch.FloatTensor,
        units_gen=False,
        **kwargs,
    ):
        (
            inputs_embeds,
            attention_mask,
            _,
            _
        ) = self.prepare_inputs_labels_for_llm(
            input_ids,
            attention_mask,
            None,
            speech_values,
            speech_mask,
            None,
            left_pad=True,
            inference=True
        )
        if not units_gen:
            generation_config = kwargs.get("generation_config", None)
            return self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
        else:
            assert len(input_ids) == 1, "only support batch size = 1"

            generation_config = kwargs.get("generation_config", None)
            tts_generation_config = kwargs.get("tts_generation_config", None)

            if generation_config is not None:
                max_new_tokens = generation_config.max_new_tokens
                if max_new_tokens is None:
                    max_new_tokens = 512
                eos_token_id = generation_config.eos_token_id
            else:
                max_new_tokens = kwargs.get("max_new_tokens", 512)
                eos_token_id = kwargs.get("eos_token_id", 0)
                generation_config = GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    eos_token_id=eos_token_id,
                    do_sample=False
                )
            if tts_generation_config is not None:
                tts_max_new_tokens = tts_generation_config.max_new_tokens
                if tts_max_new_tokens is None:
                    tts_max_new_tokens = 512
                tts_eos_token_id = tts_generation_config.eos_token_id
            else:
                tts_max_new_tokens = kwargs.get("tts_max_new_tokens", 512)
                tts_eos_token_id = tts_generation_config.get("eos_token_id", 0)
                tts_generation_config = GenerationConfig(
                    max_new_tokens=tts_max_new_tokens,
                    eos_token_id=tts_eos_token_id,
                    do_sample=False
                )
            

            streamer = kwargs.get("streamer", None)
            units_streamer = kwargs.get("units_streamer", None)

            spk_emb = kwargs.get("spk_emb", None)
            if spk_emb is None:
                spk_emb = torch.zeros(1,1,512).to(inputs_embeds.device).to(inputs_embeds.dtype)
            else:
                spk_emb = torch.from_numpy(spk_emb).to(inputs_embeds.device).to(inputs_embeds.dtype)
            spk_emb = self.tts_lm_model.spk_ln(self.tts_lm_model.spk2tts(spk_emb))

            sequences = []
            units = []
            tts_inputs_embeds = [spk_emb]
            llm_past_key_values = None
            tts_past_key_values = None


            is_finished = False
            first_step = True
            while not is_finished:
                _max_new_tokens = min(self.text_span, max_new_tokens)
                generation_config.update(
                    **{
                        "max_new_tokens": _max_new_tokens
                    }
                )
                llm_output = self.llm_model.generate(
                    inputs_embeds=inputs_embeds,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    past_key_values=llm_past_key_values,
                )

                llm_past_key_values = llm_output.past_key_values
                generated_length = len(llm_output.sequences[0])
                max_new_tokens -= generated_length
                if llm_output.sequences[0].tolist()[-1] in eos_token_id or max_new_tokens < 1:
                    is_finished = True
                if streamer is not None:
                    for token_id in llm_output.sequences[0]:
                        streamer.put(token_id.unsqueeze(0))
                sequences.append(llm_output.sequences)
                inputs_embeds = torch.cat([
                    inputs_embeds,
                    self.llm_model.get_input_embeddings()(llm_output.sequences)
                ], dim=1)

                hidden_states = llm_output.hidden_states
                if first_step:
                    hidden_states = torch.cat([hidden_states[0][-1][:, -1:, :]] + [hidden_states[i][-1] for i in range(len(hidden_states)-generated_length+1, len(hidden_states))], dim=1)
                    first_step = False
                else:
                    hidden_states = torch.cat([hidden_states[i][-1] for i in range(len(hidden_states)-generated_length, len(hidden_states))], dim=1)
                hidden_states = self.llm_ln(self.llm2tts(hidden_states))

                tts_inputs_embeds.append(hidden_states)
                if is_finished:
                    _tts_max_new_tokens = tts_max_new_tokens
                else:
                    _tts_max_new_tokens = min(self.speech_span, tts_max_new_tokens)
                if _tts_max_new_tokens < 1:
                    break
                tts_generation_config.update(
                    **{
                        "max_new_tokens": _tts_max_new_tokens
                    }
                )
                if is_finished:
                    tts_output = self.tts_lm_model.generate(
                        inputs_embeds=torch.cat(tts_inputs_embeds, dim=1),
                        generation_config=tts_generation_config,
                        past_key_values=tts_past_key_values,
                        return_dict_in_generate=True,
                        streamer=units_streamer,
                    )
                else:
                    tts_output = self.tts_lm_model.generate(
                        inputs_embeds=torch.cat(tts_inputs_embeds, dim=1),
                        generation_config=tts_generation_config,
                        past_key_values=tts_past_key_values,
                        return_dict_in_generate=True,
                    )
                    if units_streamer is not None:
                        for token_id in tts_output.sequences[0]:
                            units_streamer.put(token_id.unsqueeze(0))

                tts_past_key_values = tts_output.past_key_values
                tts_generated_length = len(tts_output.sequences[0])
                tts_max_new_tokens -= tts_generated_length
                units.append(tts_output.sequences)
                tts_inputs_embeds.append(
                    self.tts_lm_model.get_input_embeddings()(tts_output.sequences)
                )

            sequences = torch.cat(sequences, dim=1)
            units = torch.cat(units, dim=1)

            if streamer is not None:
                streamer.end()
            if units_streamer is not None:
                units_streamer.end()

            return {
                "sequences": sequences,
                "units": units,
            }