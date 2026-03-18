import os  
import sys
import uuid
from threading import Thread
from queue import Queue

import torchaudio
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, WhisperFeatureExtractor
from transformers.generation.streamers import BaseStreamer

sys.path.insert(0, "/share/workspace/EQ-SLM/EQ-Bench/src/03-eval-slm/models/GLM-4-Voice")

from speech_tokenizer.modeling_whisper import WhisperVQEncoder

sys.path.insert(0, "/share/workspace/EQ-SLM/EQ-Bench/src/03-eval-slm/models/GLM-4-Voice/cosyvoice")
sys.path.insert(0, "/share/workspace/EQ-SLM/EQ-Bench/src/03-eval-slm/models/GLM-4-Voice/third_party/Matcha-TTS")

from speech_tokenizer.utils import extract_speech_token
from flow_inference import AudioDecoder
from audio_process import AudioStreamProcessor

class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
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
        return value

class ModelWorker:
    def __init__(self, model_path, dtype="bfloat16", device='cuda'):
        self.device = device
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ) if dtype == "int4" else None

        self.glm_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=self.bnb_config if self.bnb_config else None,
            device_map={"": 0}
        ).eval()
        self.glm_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    @torch.inference_mode()
    def generate_stream(self, params):
        inputs = self.glm_tokenizer([params["prompt"]], return_tensors="pt").to(self.device)
        streamer = TokenStreamer(skip_prompt=True)
        thread = Thread(
            target=self.glm_model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=int(params.get("max_new_tokens", 256)),
                temperature=float(params.get("temperature", 1.0)),
                top_p=float(params.get("top_p", 1.0)),
                streamer=streamer
            )
        )
        thread.start()
        for token_id in streamer:
            yield token_id

class GLM4VoiceChat:
    def __init__(self, args):
        self.args = args
        self.model_path = getattr(args, 'model_path', "/share/workspace/shared_models/01_LLM-models/glm-4-voice-9b")
        self.flow_path = getattr(args, 'flow_path', "/share/workspace/shared_models/01_LLM-models/glm-4-voice-decoder")
        self.tokenizer_path = getattr(args, 'tokenizer_path', "/share/workspace/shared_models/01_LLM-models/glm-4-voice-tokenizer")
        self.device = getattr(args, 'device', 'cuda')
        self.dtype = getattr(args, 'dtype', 'bfloat16')
        
        print("加载GLM-4-Voice模型...")
        self.model_worker = ModelWorker(self.model_path, self.dtype, self.device)
        
        self.audio_decoder = AudioDecoder(
            config_path=os.path.join(self.flow_path, "config.yaml"),
            flow_ckpt_path=os.path.join(self.flow_path, 'flow.pt'),
            hift_ckpt_path=os.path.join(self.flow_path, 'hift.pt'),
            device=self.device
        )
        
        self.whisper_model = WhisperVQEncoder.from_pretrained(self.tokenizer_path).eval().to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.tokenizer_path)

    def generate_audio(self, audio, output_file, system_prompt, user_instruction=None, max_new_tokens=2048):
        try:
            torch.cuda.empty_cache()  # 清理显存
            # 处理输入
            audio_tokens = extract_speech_token(self.whisper_model, self.feature_extractor, [audio])[0]
            if len(audio_tokens) == 0:
                raise ValueError("未能从音频中提取到tokens")
            
            audio_tokens_str = "<|begin_of_audio|>" + "".join([f"<|audio_{x}|>" for x in audio_tokens]) + "<|end_of_audio|>"
            
            if user_instruction:
                user_input = f"{audio_tokens_str}\n{user_instruction}"
                default_prompt = "User will provide you with both speech and text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
            else:
                user_input = audio_tokens_str
                default_prompt = "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
            
            if system_prompt == "":
                system_prompt = default_prompt
            
            inputs = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
            
            # 推理
            params = {"prompt": inputs, "temperature": 0.2, "top_p": 0.8, "max_new_tokens": max_new_tokens}
            
            with torch.no_grad():
                text_tokens, audio_tokens = [], []
                audio_offset = self.model_worker.glm_tokenizer.convert_tokens_to_ids('<|audio_0|>')
                end_token_id = self.model_worker.glm_tokenizer.convert_tokens_to_ids('<|user|>')
                complete_tokens = []
                prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
                flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
                this_uuid = str(uuid.uuid4())
                tts_speechs = []
                tts_mels = []
                prev_mel = None
                is_finalize = False
                block_size_list = [25, 50, 100, 150, 200]
                block_size_idx = 0
                block_size = block_size_list[block_size_idx]
                audio_processor = AudioStreamProcessor()
                
                for token_id in self.model_worker.generate_stream(params):
                    if token_id == end_token_id:
                        is_finalize = True
                        
                    if len(audio_tokens) >= block_size or (is_finalize and audio_tokens):
                        if block_size_idx < len(block_size_list) - 1:
                            block_size_idx += 1
                            block_size = block_size_list[block_size_idx]
                        
                        tts_token = torch.tensor(audio_tokens, device=self.device).unsqueeze(0)
                        
                        if prev_mel is not None:
                            prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                        
                        tts_speech, tts_mel = self.audio_decoder.token2wav(
                            tts_token, 
                            uuid=this_uuid,
                            prompt_token=flow_prompt_speech_token.to(self.device),
                            prompt_feat=prompt_speech_feat.to(self.device),
                            finalize=is_finalize
                        )
                        prev_mel = tts_mel
                        
                        tts_speechs.append(tts_speech.squeeze())
                        tts_mels.append(tts_mel)
                        flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                        audio_tokens = []
                    
                    if not is_finalize:
                        complete_tokens.append(token_id)
                        if token_id >= audio_offset:
                            audio_tokens.append(token_id - audio_offset)
                        else:
                            text_tokens.append(token_id)
            
            # 保存音频
            if tts_speechs:
                tts_speech = torch.cat(tts_speechs, dim=-1).cpu().squeeze(0)  # 确保是2D张量
            else:
                tts_speech = torch.zeros(1, 1)
                
            text_only = self.model_worker.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torchaudio.save(output_file, tts_speech.unsqueeze(0), 22050, format="wav")
            
            return {
                "response_audio_transcript": text_only,
                "response_text": "",
            }
            
        except Exception as e:
            print(f"error: {audio}")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            empty_audio = torch.zeros(1, 1)
            torchaudio.save(output_file, empty_audio.unsqueeze(0), 22050, format="wav")
            return {
                "response_audio_transcript": "",
                "response_text": "",
            }         