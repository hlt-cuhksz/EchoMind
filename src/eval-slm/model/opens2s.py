import torch
import torchaudio
import numpy as np
import base64
import tempfile
import uuid
import os
import json
from io import BytesIO
from abc import ABC, abstractmethod
from backoff import on_exception, expo
from threading import Thread
from queue import Queue
from transformers import TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer
from copy import deepcopy

import sys
sys.path.insert(0, "echomind-master/src/eval-slm/models/OpenS2S")

class GlobalArgs:
    def __init__(self, model_path, flow_path):
        self.model_path = model_path
        self.flow_path = flow_path

from model_worker import load_pretrained_model, load_flow_model
from src.constants import DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN
from src.constants import DEFAULT_AUDIO_TOKEN, AUDIO_TOKEN_INDEX
from src.utils import get_waveform
from src.feature_extraction_audio import WhisperFeatureExtractor


class BaseS2SModel(ABC):
    def __init__(self, args):
        pass
    
    @abstractmethod
    @on_exception(expo, Exception, max_tries=3)
    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt=None,
        max_new_tokens=512,
        text_prompt=None
    ):
        
        pass


class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TokenStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


class OmniSpeechS2S(BaseS2SModel):
    
    def __init__(self, args):
        super().__init__(args)
        
        if hasattr(args, 'model_path'):
            model_path = args.model_path
        else:
            model_path = "OpenS2S"  #your CASIA-LM/OpenS2S model path
        
        if hasattr(args, 'flow_path'):
            flow_path = args.flow_path
        else:
            flow_path = "glm-4-voice-decoder" #your glm-4-voice-decoder model path
        
        import sys
        model_worker_module = sys.modules.get('model_worker')
        if model_worker_module:
            model_worker_module.args = GlobalArgs(model_path, flow_path)
        
        print("Loading models...")
        self.tokenizer, self.tts_tokenizer, self.model, self.generation_config, \
            self.tts_generation_config = load_pretrained_model(model_path)
        
        self.audio_extractor = WhisperFeatureExtractor.from_pretrained(
            os.path.join(model_path, "audio"))
        self.audio_decoder = load_flow_model(flow_path)
        
        self.units_bias = self.tts_tokenizer.encode("<|audio_0|>")[0]
        
        print("OmniSpeech model initialized successfully!")

    def _construct_standard_messages(self, audio_path, text_prompt=None):
        user_content = []
        
        if text_prompt and text_prompt.strip():
            user_content.append({
                "text": text_prompt.strip(),
                "audio": "",
                "speech_units": "",
                "spk_emb": ""
            })
        
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as audio_file:
                audio_binary = audio_file.read()
            audio_base64 = base64.b64encode(audio_binary).decode("utf-8")
            
            user_content.append({
                "text": "",
                "audio": audio_base64,
                "speech_units": "",
                "spk_emb": ""
            })
        
        if not user_content:
            raise ValueError("At least one of audio_path or text_prompt must be provided")
        
        messages = [
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return messages

    def _get_input_params(self, messages):

        new_messages = []
        audios = []
        

        for turn in messages:
            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                new_content = ""
                for item in content:
                    if item.get("audio", ""):
                        try:
                            audio_binary = base64.b64decode(item["audio"])
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                                temp_file.write(audio_binary)
                                temp_file_path = temp_file.name
                                waveform = get_waveform(temp_file_path)
                                audios.append(waveform)
                                print(f"Successfully loaded audio with shape: {waveform.shape}")
                        except Exception as e:
                            print(f"Error processing audio: {e}")
                            continue
                        new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                    elif item.get("text", ""):
                        new_content += item["text"]
            elif isinstance(content, dict):
                new_content = ""
                if content.get("audio", ""):
                    try:
                        audio_binary = base64.b64decode(content["audio"])
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                            temp_file.write(audio_binary)
                            temp_file_path = temp_file.name
                            waveform = get_waveform(temp_file_path)
                            audios.append(waveform)
                            print(f"Successfully loaded audio with shape: {waveform.shape}")
                    except Exception as e:
                        print(f"Error processing audio: {e}")
                        continue
                    new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                elif content.get("text", ""):
                    new_content += content["text"]
            else:
                raise NotImplementedError
            new_messages.append({"role": f"{role}", "content": f"{new_content}"})

        prompt = self.tokenizer.apply_chat_template(
            new_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompt += DEFAULT_TTS_START_TOKEN
        segments = prompt.split(f"{DEFAULT_AUDIO_TOKEN}")
        input_ids = []
        for idx, segment in enumerate(segments):
            if idx != 0:
                input_ids += [AUDIO_TOKEN_INDEX]
            input_ids += self.tokenizer.encode(segment)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)

        print(f"Number of audio inputs: {len(audios)}")
        print(f"Input sequence length: {input_ids.shape[1]}")
        
        if audios:
            try:
                speech_inputs = self.audio_extractor(
                    audios,
                    sampling_rate=self.audio_extractor.sampling_rate,
                    return_attention_mask=True,
                    return_tensors="pt"
                )
                speech_values = speech_inputs.input_features
                speech_mask = speech_inputs.attention_mask
                print(f"Speech features shape: {speech_values.shape}")
                print(f"Speech mask shape: {speech_mask.shape}")
            except Exception as e:
                print(f"Error extracting speech features: {e}")
                speech_values, speech_mask = None, None
        else:
            speech_values, speech_mask = None, None
            print("No audio inputs - using text-only mode")
        
        return input_ids, speech_values, speech_mask

    @torch.inference_mode()
    @on_exception(expo, Exception, max_tries=3)
    def generate_audio(
        self,
        audio,
        output_file,
        text_prompt=None,  
        max_new_tokens=512,
        user_instruction=None,
        audio_output=None  
    ):
       
        print(f"Processing request:")
        print(f"  - Audio: {audio}")
        if isinstance(text_prompt, tuple):
            text_prompt = " ".join(str(item) for item in text_prompt) if text_prompt else ""
        text_prompt = user_instruction#only use user_prompt
        print(f"  - Text prompt: {text_prompt}")
        # print(f"  - System prompt (using internal fixed): {self.system_prompt}")
        print(f"  - Output: {output_file}")
        
        try:
            messages = self._construct_standard_messages(audio, text_prompt)
            
            print("Constructed standard messages format:")
            # print(json.dumps(messages, indent=2, ensure_ascii=False))
            
            input_ids, speech_values, speech_mask = self._get_input_params(messages)
            input_ids = input_ids.to(device='cuda', non_blocking=True)
            
            if speech_values is not None:
                speech_values = speech_values.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
                speech_mask = speech_mask.to(device='cuda', non_blocking=True)
            
            generation_config = deepcopy(self.generation_config)
            tts_generation_config = deepcopy(self.tts_generation_config)
            
            generation_config.update(
                **{
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": True,
                }
            )
            
            tts_generation_config.update(
                **{
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0
                }
            )
            
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=False, skip_special_tokens=True, timeout=15)
            units_streamer = TokenStreamer(skip_prompt=False, timeout=15)
            
            thread = Thread(target=self.model.generate, kwargs=dict(
                input_ids=input_ids,
                attention_mask=None,
                speech_values=speech_values,
                speech_mask=speech_mask,
                spk_emb=None,
                units_gen=True,
                streamer=streamer,
                units_streamer=units_streamer,
                generation_config=generation_config,
                tts_generation_config=tts_generation_config,
                use_cache=True,
            ))
            thread.start()
            
            generated_text = ""
            all_audio_chunks = []
            units = []
            this_uuid = uuid.uuid4()
            prompt_speech_feat = torch.zeros(1, 0, 80).to(device='cuda')
            flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device='cuda')
            tts_mels = []
            prev_mel = None
            block_size = 24
            
            iter_text = iter(streamer)
            iter_units = iter(units_streamer)
            active_text = True
            active_units = True
            
            while active_text or active_units:
                if active_text:
                    try:
                        new_text = next(iter_text)
                        generated_text += new_text
                    except StopIteration:
                        active_text = False
                        print("Text generation finished")
                
                if active_units:
                    try:
                        new_unit = next(iter_units)
                        units.append(new_unit - self.units_bias)
                        
                        if len(units) >= block_size:
                            tts_token = torch.LongTensor(units).unsqueeze(0).to(device='cuda')
                            
                            if prev_mel is not None and len(tts_mels) > 0:
                                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                            else:
                                prompt_speech_feat = prompt_speech_feat  
                            
                            tts_speech, tts_mel = self.audio_decoder.token2wav(
                                tts_token, 
                                uuid=this_uuid,
                                prompt_token=flow_prompt_speech_token.to(device='cuda'),
                                prompt_feat=prompt_speech_feat,
                                finalize=False
                            )
                            
                            prev_mel = tts_mel
                            tts_mels.append(tts_mel)
                            generated_audio = tts_speech.cpu()
                            all_audio_chunks.append(generated_audio)
                            
                            flow_prompt_speech_token = torch.cat(
                                (flow_prompt_speech_token, tts_token), dim=-1)
                            units = []
                            print(f"Generated audio chunk {len(all_audio_chunks)}")
                            
                    except StopIteration:
                        active_units = False
                        print("Audio units generation finished")
                        
                        if units:
                            tts_token = torch.LongTensor(units).unsqueeze(0).to(device='cuda')
                            
                            if prev_mel is not None and len(tts_mels) > 0:
                                prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                            else:
                                prompt_speech_feat = prompt_speech_feat  
                            
                            tts_speech, tts_mel = self.audio_decoder.token2wav(
                                tts_token, 
                                uuid=this_uuid,
                                prompt_token=flow_prompt_speech_token.to(device='cuda'),
                                prompt_feat=prompt_speech_feat,
                                finalize=True
                            )
                            
                            generated_audio = tts_speech.cpu()
                            all_audio_chunks.append(generated_audio)
                            print(f"Generated final audio chunk")
            
            thread.join()
            
            if all_audio_chunks:
                output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "."
                os.makedirs(output_dir, exist_ok=True)
                
                final_audio = torch.cat(all_audio_chunks, dim=-1)
                torchaudio.save(output_file, final_audio, 22050, format="wav")
                print(f"Audio saved to: {output_file}")
                
                return {
                    'response_audio_transcript': generated_text,
                    'response_audio_path': output_file,
                    'success': True
                }
            else:
                print("No audio generated")
                return {
                    'response_audio_transcript': generated_text,
                    'response_audio_path': None,
                    'success': False,
                    'error': 'No audio generated'
                }
                
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return {
                'response_audio_transcript': f"Generation failed: {str(e)}",
                'response_audio_path': None,
                'success': False,
                'error': str(e)
            }
