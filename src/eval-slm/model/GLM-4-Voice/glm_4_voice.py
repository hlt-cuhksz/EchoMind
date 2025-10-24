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

# 导入GLM4VoiceInference类 (假设它在同一个文件或模块中)
# from your_glm4_voice_inference_module import GLM4VoiceInference

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
        
        # 初始化GLM4VoiceInference实例来替代原来的组件初始化
        self.inference_engine = GLM4VoiceInference(
            model_path=self.model_path,
            flow_path=self.flow_path,
            tokenizer_path=self.tokenizer_path,
            device=self.device,
            dtype=self.dtype
        )
        
        print("GLM-4-Voice模型加载完成!")

    def generate_audio(self, audio, output_file, system_prompt, user_instruction=None, max_new_tokens=2048):
        """
        使用GLM4VoiceInference来生成音频和文本
        
        Args:
            audio: 输入音频文件路径
            output_file: 输出音频文件路径
            system_prompt: 系统提示词
            user_instruction: 用户指令文本（可选）
            max_new_tokens: 最大新token数
            
        Returns:
            dict: 包含response_audio_transcript和response_text的字典
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 准备输入参数
            audio_path = audio
            text_input = user_instruction if user_instruction else None
            
            # 使用临时目录作为输出目录
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                # 调用GLM4VoiceInference的推理方法
                result = self.inference_engine.inference(
                    audio_path=audio_path,
                    text_input=text_input,
                    output_dir=temp_dir,
                    temperature=0.2,
                    top_p=0.8,
                    max_new_tokens=max_new_tokens,
                    custom_system_prompt=system_prompt if system_prompt else None
                )

                
                
                # 将生成的音频文件复制到指定的输出路径
                import shutil
                if os.path.exists(result['audio_path']):
                    shutil.copy2(result['audio_path'], output_file)
                else:
                    # 如果没有生成音频，创建空音频文件
                    empty_audio = torch.zeros(1, 1)
                    torchaudio.save(output_file, empty_audio.unsqueeze(0), 22050, format="wav")
                
                # 返回结果，格式与原来的接口保持一致
                return {
                    "response_audio_transcript": result.get('text_only', ''),
                    "response_text": "",  # 保持原接口格式
                }
                
        except Exception as e:
            print(f"Error processing audio: {audio}")
            print(f"Error details: {str(e)}")
            
            # 创建空音频文件作为fallback
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            empty_audio = torch.zeros(1, 1)
            torchaudio.save(output_file, empty_audio.unsqueeze(0), 22050, format="wav")
            
            return {
                "response_audio_transcript": "",
                "response_text": "",
            }


# GLM4VoiceInference类的完整实现（从你提供的代码中复制）
import json
import tempfile
import re
from argparse import ArgumentParser

audio_token_pattern = re.compile(r"<\|audio_(\d+)\|>")

class GLM4VoiceInference:
    def __init__(self, 
                 model_path="/share/workspace/shared_models/01_LLM-models/glm-4-voice-9b",
                 flow_path="/share/workspace/shared_models/01_LLM-models/glm-4-voice-decoder",
                 tokenizer_path="/share/workspace/shared_models/01_LLM-models/glm-4-voice-tokenizer",
                 device="cuda",
                 dtype="bfloat16"):
        """
        初始化GLM-4-Voice推理器
        
        Args:
            model_path: GLM模型路径
            flow_path: Flow模型路径  
            tokenizer_path: Tokenizer路径
            device: 设备(cuda/cpu)
            dtype: 数据类型(bfloat16/int4)
        """
        self.device = device
        self.flow_path = flow_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.dtype = dtype
        
        # 初始化组件
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化所有模型组件"""
        print("正在初始化所有模型组件...")
        
        # 初始化GLM模型和tokenizer
        self.model_worker = ModelWorker(self.model_path, self.dtype, self.device)
        
        # Flow & Hift
        flow_config = os.path.join(self.flow_path, "config.yaml")
        flow_checkpoint = os.path.join(self.flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(self.flow_path, 'hift.pt')
        
        print("正在加载音频解码器...")
        self.audio_decoder = AudioDecoder(
            config_path=flow_config, 
            flow_ckpt_path=flow_checkpoint,
            hift_ckpt_path=hift_checkpoint,
            device=self.device
        )
        
        # Speech tokenizer
        print("正在加载语音tokenizer...")
        self.whisper_model = WhisperVQEncoder.from_pretrained(
            self.tokenizer_path
        ).eval().to(self.device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.tokenizer_path
        )
        
        print("所有模型组件初始化完成!")
    
    def inference(self, 
                  audio_path=None,
                  text_input=None, 
                  output_dir="./outputs",
                  temperature=0.2,
                  top_p=0.8,
                  max_new_tokens=2000,
                  custom_system_prompt=None):
        """
        执行推理
        
        Args:
            audio_path: 输入音频文件路径(可以与text_input同时使用)
            text_input: 输入文本(可以与audio_path同时使用)
            output_dir: 输出目录
            temperature: 温度参数
            top_p: top_p参数  
            max_new_tokens: 最大新token数
            custom_system_prompt: 自定义系统prompt
            
        Returns:
            dict: 包含输出音频路径和文本的字典
        """
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 检查输入
        if audio_path is None and text_input is None:
            raise ValueError("必须提供audio_path或text_input之一，或者两者都提供")
        
        # 确定输入模式和处理输入
        user_input_parts = []
        
        if audio_path is not None and text_input is not None:
            # 多模态输入：同时包含音频和文本
            input_mode = "multimodal"
            print(f"处理多模态输入:")
            print(f"  音频: {audio_path}")
            print(f"  文本: {text_input}")
            
            # 提取音频tokens
            audio_tokens = extract_speech_token(
                self.whisper_model, self.feature_extractor, [audio_path]
            )[0]
            if len(audio_tokens) == 0:
                raise ValueError("未能从音频中提取到tokens")
            audio_tokens_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens_str = "<|begin_of_audio|>" + audio_tokens_str + "<|end_of_audio|>"
            
            # 组合音频和文本输入
            user_input_parts.append(audio_tokens_str)
            # user_input_parts.append(text_input)
            user_input = "\n".join(user_input_parts)
            
            # default_system_prompt = "User will provide you with both speech and text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
            default_system_prompt = text_input
            
        elif audio_path is not None:
            # 仅音频输入
            input_mode = "audio"
            print(f"处理音频输入: {audio_path}")
            # 提取音频tokens
            audio_tokens = extract_speech_token(
                self.whisper_model, self.feature_extractor, [audio_path]
            )[0]
            if len(audio_tokens) == 0:
                raise ValueError("未能从音频中提取到tokens")
            audio_tokens_str = "".join([f"<|audio_{x}|>" for x in audio_tokens])
            audio_tokens_str = "<|begin_of_audio|>" + audio_tokens_str + "<|end_of_audio|>"
            user_input = audio_tokens_str
            
            default_system_prompt = " "
        # else:
        #     # 仅文本输入
        #     input_mode = "text"  
        #     user_input = text_input
        #     print(f"处理文本输入: {text_input}")
        #     default_system_prompt = "User will provide you with a text instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens."
        
        # 使用自定义系统prompt或默认prompt
        system_prompt = custom_system_prompt if custom_system_prompt else default_system_prompt
        
        print(f"system_prompt: {system_prompt}")


        # 构建输入
        inputs = f"<|system|>\n{system_prompt}<|user|>\n{user_input}<|assistant|>streaming_transcription\n"
        
        print(f"开始推理，输入模式: {input_mode}")
        
        # 准备推理参数
        params = {
            "prompt": inputs,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
        }
        
        # 执行推理
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
            
            print("正在生成音频流...")
            
            # 直接调用模型生成流
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
        
        # 合并所有音频片段
        if tts_speechs:
            tts_speech = torch.cat(tts_speechs, dim=-1).cpu()
        else:
            tts_speech = torch.zeros(1, 1)
            
        # 解码文本
        complete_text = self.model_worker.glm_tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
        text_only = self.model_worker.glm_tokenizer.decode(text_tokens, ignore_special_tokens=False)
        
        # 保存输出
        timestamp = uuid.uuid4().hex[:8]
        
        # 保存音频
        audio_output_path = os.path.join(output_dir, f"output_audio_{timestamp}.wav")
        torchaudio.save(audio_output_path, tts_speech.unsqueeze(0), 22050, format="wav")
        
        # 保存文本
        text_output_path = os.path.join(output_dir, f"output_text_{timestamp}.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            f.write(f"完整输出:\n{complete_text}\n\n")
            f.write(f"纯文本部分:\n{text_only}\n\n")
            f.write(f"输入模式: {input_mode}\n")
            if input_mode == "audio":
                f.write(f"输入音频: {audio_path}\n")
            else:
                f.write(f"输入文本: {text_input}\n")
            f.write(f"推理参数:\n")
            f.write(f"  temperature: {temperature}\n")
            f.write(f"  top_p: {top_p}\n")
            f.write(f"  max_new_tokens: {max_new_tokens}\n")
        
        print(f"推理完成!")
        print(f"音频输出: {audio_output_path}")
        print(f"文本输出: {text_output_path}")
        print(f"生成文本预览: {text_only[:100]}...")
        
        return {
            "audio_path": audio_output_path,
            "text_path": text_output_path,
            "complete_text": complete_text,
            "text_only": text_only
        }