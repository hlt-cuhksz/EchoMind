import argparse
import itertools
import json, ujson
import os, sys
import random
import time
from functools import partial
import re

sys.path.append('echomind-master/src/eval-slm/models/Baichuan-Omni-1.5/web_demo')

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import editdistance as ed
from tqdm import tqdm
import time
import torchaudio
torch.set_num_threads(1)


class ASRModel:
    def __init__(self, args):
        self.args = args
        
        self.max_new_tokens = getattr(args, 'max_new_tokens', 700)
        self.do_sample = getattr(args, 'do_sample', False)
        self.num_beams = getattr(args, 'num_beams', 1)
        self.top_k = getattr(args, 'top_k', 5)
        self.top_p = getattr(args, 'top_p', 0.85)
        self.temperature = getattr(args, 'temperature', 0.5)
        self.repetition_penalty = getattr(args, 'repetition_penalty', 1.3)
        self.checkpoint = getattr(args, 'checkpoint', 'Baichuan-Omni-1d5')#your Baichuan-Omni-1d5 model path

        
        self.model, self.tokenizer = self._load_model_tokenizer()
        self.model.bind_processor(self.tokenizer, training=False, relative_path='/')
        
        audio_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_start_token_id)
        audio_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_end_token_id)
        self.PROMPT_ASR = '将语音转录为文本:' + audio_start_token + '{}' + audio_end_token
        
    def _load_model_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, 
            trust_remote_code=True,
            model_max_length=128000,
        )
        device_map = 'cuda'
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        model.config.use_cache = True
        return model, tokenizer

    def generate_audio(self, audio_file, output_file, text_prompt, user_instruction):
        try:
            batch_data = [{
                'audio': audio_file,
                'uttid': os.path.basename(audio_file).split('.')[0] 
            }]
            
            prompts = []
            for b in batch_data:
                prompt = self.PROMPT_ASR.format(json.dumps({'path': b['audio']}))
                prompts.append(prompt)
            
            ret = self.model.processor(prompts)
            
            predicted_ids = self.model.generate(
                input_ids=ret.input_ids.cuda(),
                attention_mask=ret.attention_mask.cuda(),
                labels=None,
                audios=ret.audios.cuda() if ret.audios is not None else None,
                encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
                bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=self.do_sample,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                num_return_sequences=1,
                repetition_penalty=self.repetition_penalty,
            )
            
            generated = self.tokenizer.batch_decode(
                predicted_ids[:, ret.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            result = {
                'response_audio_transcript': generated[0].strip(),
                'audio_file': audio_file,
                'uttid': batch_data[0]['uttid']
            }
            
            print(f'ASR Result for {batch_data[0]["uttid"]}: {generated[0].strip()}')
            
            return result
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return {
                'response_audio_transcript': '',
                'audio_file': audio_file,
                'error': str(e)
            }

 

