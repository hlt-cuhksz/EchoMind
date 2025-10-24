import argparse
import torch
import os
import json
import whisper
import sys
import uuid
import threading
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from collections import defaultdict

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

current_dir = os.path.dirname(os.path.abspath(__file__))

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

sys.path.insert(0, os.path.join(current_dir, "LLaMA-Omni2"))


from llama_omni2.model import *
from llama_omni2.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN

# 添加CosyVoice相关导入
try:
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from cosyvoice.utils.file_utils import load_wav
except ImportError:
    print("Warning: CosyVoice not available, speech synthesis will not work")
    CosyVoiceFrontEnd = None
    load_wav = None


def load_speech(path):
    """加载并预处理音频文件"""
    speech = whisper.load_audio(path)
    speech = whisper.pad_or_trim(speech)
    speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
    return speech


def process_messages(messages, tokenizer):
    """处理消息并转换为input_ids"""
    # assert len(messages) % 2 == 1, "Number of history messages must be odd"
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
    input_ids[input_ids == tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)] = SPEECH_TOKEN_INDEX
    return input_ids


def load_pretrained_model(model_path, s2s=True):
    model_cls = Omni2Speech2SQwen2ForCausalLM if s2s else Omni2SpeechQwen2ForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    config.tts_tokenizer = os.path.join(model_path, "tts_tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = model_cls.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model.cuda()
    return tokenizer, model


def generate_response(
    audio_path,
    tokenizer,
    model,
    system_prompt="",
    user_instruction=None,
    temperature=0,
    top_p=None,
    top_k=None,
    num_beams=1,
    max_new_tokens=256
):
    """
    生成单个音频的响应
    
    Args:
        audio_path: 输入音频文件路径
        system_prompt: 系统提示词
        user_instruction: 用户文本（可选）
        model_path: 模型路径
        temperature: 生成温度
        top_p: 核心采样参数
        top_k: 选择前k个最可能的token
        num_beams: 束搜索大小
        max_new_tokens: 最大生成token数
    
    Returns:
        dict: 包含prediction和prediction_units（如果是s2s模式）的字典
    """
    
    # 加载和处理音频
    speech = load_speech(audio_path)
    
    # 构建消息
    if user_instruction:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": DEFAULT_SPEECH_TOKEN},
            {"role": "assistant", "content": user_instruction}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": DEFAULT_SPEECH_TOKEN}
        ]
    
    # 处理消息
    input_ids = process_messages(messages, tokenizer)
    
    # 准备输入数据
    input_ids = input_ids.unsqueeze(0).to(device='cuda', non_blocking=True)
    speech_tensor = speech.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
    speech_lengths = torch.LongTensor([len(speech)]).to(device='cuda', non_blocking=True)
    
    # 生成响应
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            speech=speech_tensor,
            speech_lengths=speech_lengths,
            do_sample=True if temperature > 0 else False,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        output_ids, output_units = outputs

    # 解码输出文本
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
    result = {
        "prediction": output_text,
        "prediction_units": output_units
    }
    
    return result

def fade_in_out(fade_in_mel, fade_out_mel, window):
    device = fade_in_mel.device
    fade_in_mel, fade_out_mel = fade_in_mel.cpu(), fade_out_mel.cpu()
    mel_overlap_len = int(window.shape[0] / 2)
    fade_in_mel[..., :mel_overlap_len] = fade_in_mel[..., :mel_overlap_len] * window[:mel_overlap_len] + \
                                         fade_out_mel[..., -mel_overlap_len:] * window[mel_overlap_len:]
    return fade_in_mel.to(device)


class SpeechDecoder:
    def __init__(self,
        model_dir=os.path.join(base_dir, 'models', 'cosy2_decoder'),
        device="cuda",
        hop_len=None,
        load_jit=False,
        load_trt=False,
        load_onnx=False,
    ):
        self.device = device

        # Config
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})

        # Frontend
        self.frontend = CosyVoiceFrontEnd(
            configs['get_tokenizer'],
            configs['feat_extractor'],
            '{}/campplus.onnx'.format(model_dir),
            '{}/speech_tokenizer_v2.onnx'.format(model_dir),
            '{}/spk2info.pt'.format(model_dir),
            False,
            configs['allowed_special']
        )
        self.sample_rate = configs['sample_rate']

        # Load models
        self.flow = configs['flow']
        self.flow.load_state_dict(torch.load('{}/flow.pt'.format(model_dir), map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        self.flow.decoder.fp16 = False
        self.hift = configs['hift']
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load('{}/hift.pt'.format(model_dir), map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

        if load_jit:
            self.load_jit('{}/flow.encoder.fp32.zip'.format(model_dir))
        if load_trt is True and load_onnx is True:
            load_onnx = False
            logging.warning('can not set both load_trt and load_onnx to True, force set load_onnx to False')
        if load_onnx:
            self.load_onnx('{}/flow.decoder.estimator.fp32.onnx'.format(model_dir))
        if load_trt:
            self.load_trt('{}/flow.decoder.estimator.fp16.Volta.plan'.format(model_dir))

        self.token_hop_len = hop_len if hop_len is not None else 2 * self.flow.input_frame_rate
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # dict used to store session related variable
        self.lock = threading.Lock()
        self.hift_cache_dict = {}
    
    def load_jit(self, flow_encoder_model):
        print("Loading JIT model")
        self.flow.encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
    
    def load_onnx(self, flow_decoder_estimator_model):
        print("Loading ONNX model")
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option, providers=providers)
    
    def load_trt(self, flow_decoder_estimator_model):
        print("Loading TRT model")
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()
        self.flow.decoder.fp16 = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech
    
    def entry(self, generated_tokens, prompt_speech_16k, stream=False, speed=1.0):
        prompt_speech_feat = torch.zeros(1, 0, 80)
        prompt_speech_token = torch.zeros(1, 0, dtype=torch.int32)
        embedding = self.frontend._extract_spk_embedding(prompt_speech_16k)
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.hift_cache_dict[this_uuid] = None
        if stream:
            token_offset = 0
            this_tts_speech_output = []
            while True:
                if len(generated_tokens) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = generated_tokens[:token_offset + self.token_hop_len + self.flow.pre_lookahead_len].unsqueeze(0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=embedding,
                        uuid=this_uuid,
                        token_offset=token_offset,
                        finalize=False
                    )
                    token_offset += self.token_hop_len
                    this_tts_speech_output.append(this_tts_speech.cpu())
                else:
                    break
            this_tts_speech_token = generated_tokens.unsqueeze(0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=embedding,
                uuid=this_uuid,
                token_offset=token_offset,
                finalize=True
            )
            this_tts_speech_output.append(this_tts_speech.cpu())
            return torch.cat(this_tts_speech_output, dim=1)
        else:
            this_tts_speech_token = generated_tokens.unsqueeze(0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=embedding,
                uuid=this_uuid,
                token_offset=0,
                finalize=True,
                speed=speed
            )
            return this_tts_speech.cpu()
    

def process_units(input_str):
    import re
    numbers = re.findall(r'\d+', input_str)
    units = list(map(int, numbers))
    return units  

def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav)
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

class LlamaOmni2Chat:
    def __init__(self, args):
        self.args = args
        self.tokenizer, self.model = load_pretrained_model(os.path.join(base_dir, 'models', 'LLaMA-Omni2-7B'))

    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt="",
        user_instruction=None,
        max_new_tokens=2048,
        audio_output=False
    ):
        try:
            # 生成文本响应
            print("正在生成文本响应...")
            result = generate_response(
                audio_path=audio,
                tokenizer=self.tokenizer,
                model=self.model,
                system_prompt=system_prompt,
                user_instruction=user_instruction
            )
                  
            # 添加必要的路径
            ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.append(os.path.join(ROOT_DIR))
            sys.path.append(os.path.join(ROOT_DIR, "third_party/Matcha-TTS"))

            # 选择提示音频
            prompt_speech = os.path.join(current_dir, "LLaMA-Omni2", "llama_omni2", "inference", "prompt_en.wav")

            prompt_speech_16k = load_wav(prompt_speech, 16000)

            # 初始化语音解码器
            speech_decoder = SpeechDecoder()
            
            # 处理语音单元并生成音频
            units = process_units(str(result['prediction_units']))
            x = torch.LongTensor(units).cuda()
            tts_speech = speech_decoder.entry(
                x, 
                prompt_speech_16k
            )

            if audio_output:
                torchaudio.save(output_file, tts_speech, 24000)
                print(f"音频已保存到: {output_file}")

            print("处理完成！")

            return {
                "response_audio_transcript": result['prediction'],
                "response_text": result['prediction']
            }
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            return {
                "response_audio_transcript": "",
                "response_text": ""
            }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA-Omni2 单音频推理")
    parser.add_argument("--audio", type=str, required=True, help="输入音频文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出音频文件路径")
    parser.add_argument("--system_prompt", type=str, default="", help="系统提示词")
    parser.add_argument("--user_instruction", type=str, default="", help="用户文本（可选）")
    parser.add_argument("--temperature", type=float, default=0, help="生成温度")
    
    args = parser.parse_args()
    
    chat_model = LlamaOmni2Chat(args)
    chat_model.generate_audio(
        audio=args.audio,
        output_file=args.output_file,
        system_prompt=args.system_prompt,
        user_instruction=args.user_instruction
    )