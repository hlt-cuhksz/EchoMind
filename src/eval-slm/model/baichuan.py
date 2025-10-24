import sys
import os
import numpy as np
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.append('echomind-master/src/eval-slm/models/Baichuan-Omni-1.5/web_demo')

from generation import decode_wave_vocoder, GenerationAudioTokens
import time
import re
import json, ujson
from constants import *

class BaichuanChat:
    def __init__(self, args):
        self.args = args
        self.device = args.device if hasattr(args, 'device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        sys.path.append(os.path.join(COSY_VOCODER))
        from cosy24k_vocoder import Cosy24kVocoder
        self.vocoder = Cosy24kVocoder.from_pretrained(os.path.join(COSY_VOCODER, "hift.pt"))
        self.vocoder = self.vocoder.cuda() if torch.cuda.is_available() else self.vocoder
        
        self.model, self.tokenizer = self._init_model()
        
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_start_token_id)
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audio_end_token_id)
        self.audiogen_start_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_start_token_id)
        self.audiogen_end_token = self.tokenizer.convert_ids_to_tokens(self.model.config.audio_config.audiogen_end_token_id)
        self.special_token_pattern = re.compile('<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>')
        
        self.cache_dir = getattr(args, 'cache_dir', './cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.turn_counter = 0

    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        if torch.cuda.is_available():
            model = model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model.training = False
        model.bind_processor(tokenizer, training=False, relative_path="/")
        return model, tokenizer

    def _wave_concat(self, wave_list, start, overlap=400):
        new_wave_list = []
        cur = start
        for wave in wave_list[start:]:
            if (
                cur - 1 >= 0
                and wave_list[cur - 1].shape[1] > overlap
                and wave.shape[1] > overlap
            ):
                new_wave_list.append(
                    (
                        wave_list[cur - 1][:, -overlap:]
                        * torch.linspace(
                            1.0, 0.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                        + wave[:, :overlap]
                        * torch.linspace(
                            0.0, 1.0, overlap, device=wave_list[cur - 1].device
                        )[None, :]
                    )
                )
            new_wave_list.append(wave)
            cur += 1
        return torch.cat(new_wave_list, dim=1)

    def _save_local(self, wave, local_path):
        torchaudio.save(local_path, torch.cat(wave, dim=0).cpu(), sampling_rate)
        return self.audiogen_start_token + ujson.dumps({'path': local_path}, ensure_ascii=False) + self.audiogen_end_token

    def _generate_text_step(self, pret, plen, kv_cache_flag, audiogen_flag=True):
        if not kv_cache_flag:
            textret = self.model.generate(
                pret.input_ids.cuda() if torch.cuda.is_available() else pret.input_ids,
                attention_mask=pret.attention_mask.cuda() if torch.cuda.is_available() else pret.attention_mask, 
                audios = pret.audios.cuda() if pret.audios is not None and torch.cuda.is_available() else pret.audios,
                encoder_length=pret.encoder_length.cuda() if pret.encoder_length is not None and torch.cuda.is_available() else pret.encoder_length,
                bridge_length=pret.bridge_length.cuda() if pret.bridge_length is not None and torch.cuda.is_available() else pret.bridge_length,
                tokenizer=self.tokenizer,
                max_new_tokens=50 if audiogen_flag else 1024,
                stop_strings=[self.audiogen_start_token, '<|endoftext|>'] if audiogen_flag else ['<|endoftext|>'],
                do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1, return_dict_in_generate=True,
            )
        else:
            textret = self.model.generate(
                pret.sequences,
                attention_mask=torch.ones_like(pret.sequences),
                tokenizer=self.tokenizer,
                past_key_values=(pret.past_key_values),
                stop_strings = [self.audiogen_start_token,',','!','?','，','。','！','？','. '],
                max_new_tokens=50, do_sample=True, temperature=0.3, top_k=20, top_p=0.85, repetition_penalty=1.05, return_dict_in_generate=True,
            )
        newtext = self.tokenizer.decode(textret.sequences[0, plen:])
        return textret, newtext

    def _generate_audio_step(self, pret):
        audioret = GenerationAudioTokens.generate(
            self.model,
            pret.sequences,
            attention_mask=torch.ones_like(pret.sequences),
            past_key_values=(pret.past_key_values if pret.past_key_values is not None else None),
            max_new_tokens=500,
            do_sample=True, temperature=0.5, top_k=5, top_p=0.85, repetition_penalty=1.3, return_dict_in_generate=True,
        )
        wave_segment = decode_wave_vocoder(audioret.audios_sequences.clone(), self.vocoder, self.model)
        return audioret, wave_segment

    def _generate_response(self, content, audiogen_flag=False):
        pret = self.model.processor([content])
        plen = pret.input_ids.shape[1]
        ret, text_segment = self._generate_text_step(pret, plen, False, audiogen_flag)
        wave_list = []
        full_text = re.sub(self.special_token_pattern, '', text_segment)
        
        if audiogen_flag:
            start = 0
            for i in range(100):
                m = ret.sequences[0, -1].item()
                if m == self.tokenizer.eos_token_id:
                    if ret.sequences.shape[1] - plen > 1:
                        ret.sequences[0, -1] = (self.model.config.audio_config.audiogen_start_token_id)
                        ret, wave_segment = self._generate_audio_step(ret)
                        wave_list.extend(wave_segment)
                        audio_path = os.path.join(self.cache_dir, f'assistant_turn{self.turn_counter}_round{i}.wav')
                        full_text += self._save_local(wave_segment, audio_path)
                    break

                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_start_token_id
                ret, wave_segment = self._generate_audio_step(ret)
                wave_list.extend(wave_segment)
                audio_path = os.path.join(self.cache_dir, f'assistant_turn{self.turn_counter}_round{i}.wav')
                full_text += self._save_local(wave_segment, audio_path)

                ret.sequences[0, -1] = self.model.config.audio_config.audiogen_end_token_id
                plen = ret.sequences.shape[1]
                ret, text_segment = self._generate_text_step(ret, plen, True, True)
                full_text += re.sub(self.special_token_pattern, '', text_segment)
                print(f"ROUND {i+1}:", text_segment)

            if wave_list:
                try:
                    final_wave = self._wave_concat(wave_list, 0, overlap=wave_concat_overlap if 'wave_concat_overlap' in globals() else 400)
                    return full_text, final_wave
                except:
                    return full_text, None
            else:
                return full_text, None
        else:
            return full_text, None

    def _load_audio(self, audio_path):
        wave, sr = torchaudio.load(audio_path)
        if sr != sampling_rate:
            wave = torchaudio.functional.resample(wave, sr, sampling_rate)
        return wave

    def _parse_assistant_content(self, content):
        wave_segments = []
        text = ""

        parts = re.split(r'<audiogen_start_baichuan>', content)
        prev_text = parts[0].strip()
        text += prev_text
        
        for part in parts[1:]:
            end_split = re.split(r'<audiogen_end_baichuan>', part, 1)
            
            if len(end_split) != 2:
                continue  
                
            json_str, remaining = end_split
            json_str = json_str.strip()
            cleaned_json = json_str.replace('\\/', '/')
            
            try:
                json_data = json.loads(cleaned_json)
                if os.path.exists(json_data["path"]):
                    wave = self._load_audio(json_data["path"])
                    wave_segments.append(wave)
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                continue  
                
            text += remaining.strip()
 
        if wave_segments:
            final_wave = torch.cat(wave_segments, dim=1)
            return text, final_wave
        else:
            return text, None

    def _preprocess_messages(self, messages, audiogen_flag=True):
        text = ""
        for i, msg in enumerate(messages):
            if audiogen_flag and msg["role"] == "assistant":
                text += role_prefix['audiogen'] if 'role_prefix' in globals() else ""
            text += role_prefix[msg['role']] if 'role_prefix' in globals() else f"<{msg['role']}>: "
            text += msg['content']
        if audiogen_flag:
            text += role_prefix['audiogen'] if 'role_prefix' in globals() else ""
        text += role_prefix["assistant"] if 'role_prefix' in globals() else "<assistant>: "
        return text

    def generate_audio(self, audio_file, output_file, text_prompt, user_instruction="", audio_output=False):

        try:
            history = []
            
            if text_prompt.strip():
                history.append({
                    "role": "system", 
                    "content": text_prompt
                })
            
            fn_wav = os.path.join(self.cache_dir, f'user_turn{self.turn_counter}.wav')
            os.system(f"cp '{audio_file}' '{fn_wav}'")
            
            audio_content = self.audio_start_token + ujson.dumps({'path': fn_wav}, ensure_ascii=False) + self.audio_end_token
            if user_instruction:
                audio_content += user_instruction
            
            history.append({
                "role": "user", 
                "content": audio_content
            })
            
            content = self._preprocess_messages(history, audiogen_flag=audio_output)
            
            full_text, wave_tensor = self._generate_response(content, audiogen_flag=audio_output)
            
            response = {}
            
            if audio_output:
                if wave_tensor is not None:
                    output_dir = os.path.dirname(output_file)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                        
                    torchaudio.save(output_file, wave_tensor.cpu(), sampling_rate)
                    
                    clean_text, _ = self._parse_assistant_content(full_text)
                    response['response_audio_transcript'] = clean_text
                    response['response_text'] = clean_text
                    response['audio_file'] = output_file
                else:
                    clean_text = re.sub(self.special_token_pattern, '', full_text)
                    response['response_audio_transcript'] = clean_text
                    response['response_text'] = clean_text
                    response['audio_file'] = None
            else:
                clean_text = re.sub(self.special_token_pattern, '', full_text)
                response['response_audio_transcript'] = clean_text
                response['response_text'] = clean_text
                response['audio_file'] = None
                print(f"Text-only mode: Generated text: {clean_text[:100]}...")  
            
            self.turn_counter += 1
            return response
            
        except Exception as e:
            print(f"Error in generate_audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'response_audio_transcript': "",
                'response_text': f"Error: {str(e)}",
                'audio_file': None
            }