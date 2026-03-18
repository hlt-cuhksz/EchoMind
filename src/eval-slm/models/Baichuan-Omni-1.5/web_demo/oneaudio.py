import sys
import os
import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from generation import decode_wave_vocoder, GenerationAudioTokens
import re
import json
import ujson
from constants import *


class AudioChatSystem:
    def __init__(self, model_path=None, vocoder_path=None):

        self.model_path = model_path or MODEL_PATH
        self.vocoder_path = vocoder_path or COSY_VOCODER

        sys.path.append(os.path.join(COSY_VOCODER))
        from cosy24k_vocoder import Cosy24kVocoder
        self.vocoder = Cosy24kVocoder.from_pretrained(os.path.join(self.vocoder_path, "hift.pt"))
        self.vocoder = self.vocoder.cuda()
        
        self.model, self.tokenizer = self._init_model()
        
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_start_token_id
        )
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_end_token_id
        )
        self.audiogen_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audiogen_start_token_id
        )
        self.audiogen_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audiogen_end_token_id
        )
        
        self.history = []
        self.turn_counter = 0
        self.cache_dir = "./audio_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.special_token_pattern = re.compile(
            '<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>'
        )

    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        model.training = False
        model.bind_processor(tokenizer, training=False, relative_path="/")
        return model, tokenizer

    def _wave_concat(self, wave_list, start=0, overlap=400):
        if len(wave_list) <= start:
            return torch.empty(0, 0)
            
        new_wave_list = []
        cur = start
        
        for i, wave in enumerate(wave_list[start:], start):
            if (i > 0 and 
                wave_list[i-1].shape[1] > overlap and 
                wave.shape[1] > overlap):
                overlap_part = (
                    wave_list[i-1][:, -overlap:] * 
                    torch.linspace(1.0, 0.0, overlap, device=wave_list[i-1].device)[None, :] +
                    wave[:, :overlap] * 
                    torch.linspace(0.0, 1.0, overlap, device=wave.device)[None, :]
                )
                new_wave_list.append(overlap_part)
            new_wave_list.append(wave)
            
        return torch.cat(new_wave_list, dim=1)

    def _save_audio(self, wave_segments, filename):
        local_path = os.path.join(self.cache_dir, filename)
        if wave_segments:
            concatenated_wave = self._wave_concat(wave_segments)
            torchaudio.save(
                local_path, 
                concatenated_wave.cpu(), 
                sampling_rate
            )
        return local_path

    def _generate_text(self, processed_input, max_tokens=2048):
        result = self.model.generate(
            processed_input.input_ids.cuda(),
            attention_mask=processed_input.attention_mask.cuda(),
            audios=processed_input.audios.cuda() if processed_input.audios is not None else None,
            encoder_length=processed_input.encoder_length.cuda() if processed_input.encoder_length is not None else None,
            bridge_length=processed_input.bridge_length.cuda() if processed_input.bridge_length is not None else None,
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            stop_strings=['<|endoftext|>'],
            do_sample=True,
            temperature=0.8,
            top_k=20,
            top_p=0.85,
            repetition_penalty=1.1,
            return_dict_in_generate=True,
        )
        
        original_length = processed_input.input_ids.shape[1]
        new_text = self.tokenizer.decode(result.sequences[0, original_length:])
        clean_text = re.sub(self.special_token_pattern, '', new_text)
        
        return result, clean_text

    def _generate_audio(self, text_result):
        sequences = text_result.sequences.clone()
        sequences[0, -1] = self.model.config.audio_config.audiogen_start_token_id
        
        audio_result = GenerationAudioTokens.generate(
            self.model,
            sequences,
            attention_mask=torch.ones_like(sequences),
            past_key_values=text_result.past_key_values if hasattr(text_result, 'past_key_values') else None,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.5,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.3,
            return_dict_in_generate=True,
        )
        
        wave_segments = decode_wave_vocoder(
            audio_result.audios_sequences.clone(), 
            self.vocoder, 
            self.model
        )
        
        return wave_segments

    def _preprocess_message(self, audio_path, system_prompt):

        if len(self.history) == 0:
            self.history.append({
                "role": "system",
                "content": system_prompt
            })
        
        audio_content = (
            self.audio_start_token + 
            ujson.dumps({'path': audio_path}, ensure_ascii=False) + 
            self.audio_end_token
        )
        
        self.history.append({
            "role": "user",
            "content": audio_content
        })
        
        text = ""
        for msg in self.history:
            if msg["role"] == "system":
                text += role_prefix.get("system", "") + msg["content"]
            elif msg["role"] == "user":
                text += role_prefix.get("user", "") + msg["content"]
            elif msg["role"] == "assistant":
                text += role_prefix.get("audiogen", "") + role_prefix.get("assistant", "") + msg["content"]
        
        text += role_prefix.get("audiogen", "") + role_prefix.get("assistant", "")
        
        return text

    def generate_response(self, audio_path, system_prompt):
       
        try:
            message_text = self._preprocess_message(audio_path, system_prompt)
            processed_input = self.model.processor([message_text])
            
            text_result, generated_text = self._generate_text(processed_input)
            
            wave_segments = self._generate_audio(text_result)
            
            audio_filename = f"response_turn_{self.turn_counter}.wav"
            output_audio_path = self._save_audio(wave_segments, audio_filename)
            
            assistant_content = (
                self.audiogen_start_token + 
                ujson.dumps({'path': output_audio_path}, ensure_ascii=False) + 
                self.audiogen_end_token
            )
            
            self.history.append({
                "role": "assistant",
                "content": assistant_content
            })
            
            self.turn_counter += 1
            
            return generated_text, output_audio_path
            
        except Exception as e:
            print(f"An error occurred while generating the response: {e}")
            return f"An error occurred while generating the response: {e}", None

    def clear_history(self):
        self.history = []
        self.turn_counter = 0
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_history(self):
        return self.history.copy()


