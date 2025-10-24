"""
A model worker executes the model.
"""
import os
import argparse
import asyncio
import json
import time
import threading
import uuid
import sys
from copy import deepcopy
import base64
import tempfile
import logging
from io import BytesIO
import torchaudio

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
import numpy as np
from functools import partial

from transformers import AutoTokenizer, AutoConfig
from transformers import TextIteratorStreamer
from transformers import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from threading import Thread
from queue import Queue

from flow_inference import AudioDecoder
from src.constants import WORKER_HEART_BEAT_INTERVAL
from src.constants import DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN
from src.constants import DEFAULT_AUDIO_TOKEN, AUDIO_TOKEN_INDEX
from src.modeling_omnispeech import OmniSpeechModel
from src.utils import get_waveform
from src.feature_extraction_audio import WhisperFeatureExtractor

sys.path.append("cosyvoice")
sys.path.append("third_party/Matcha-TTS")

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
global_counter = 0
model_semaphore = None

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("worker")

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt

        # variables used in the streaming process
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
        else:
            return value


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

def load_pretrained_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tts_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tts"))
    generation_config = GenerationConfig.from_pretrained(model_path)
    tts_generation_config = GenerationConfig.from_pretrained(os.path.join(model_path, "tts"))
    model = OmniSpeechModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model.cuda()
    return tokenizer, tts_tokenizer, model, generation_config, tts_generation_config

def load_flow_model(flow_path):
    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, 'flow.pt')
    hift_checkpoint = os.path.join(args.flow_path, 'hift.pt')
    audio_decoder = AudioDecoder(flow_config, flow_checkpoint, hift_checkpoint, device="cuda")
    return audio_decoder


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register, model_path, 
                 flow_path,model_name):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name
        self.tokenizer, self.tts_tokenizer, self.model, self.generation_config,\
            self.tts_generation_config = load_pretrained_model(model_path)
        self.audio_extractor = WhisperFeatureExtractor.from_pretrained(os.path.join(model_path, "audio"))
        self.audio_decoder = load_flow_model(flow_path)
        self.system_prompt = "You are a helpful assistant."
        self.units_bias = self.tts_tokenizer.encode("<|audio_0|>")[0]

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def get_input_params(self, messages):
        new_messages = []
        audios = []
        if self.system_prompt:
            new_messages.append({"role": "system", "content": self.system_prompt})
        for turn in messages:
            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                new_content = ""
                for item in content:
                    if item.get("audio", ""):
                        audio_binary = base64.b64decode(item["audio"])
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                            temp_file.write(audio_binary)
                            temp_file_path = temp_file.name
                            waveform = get_waveform(temp_file_path)
                            audios.append(waveform)
                        new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                    elif item.get("text", ""):
                        new_content += item["text"]
            elif isinstance(content, dict):
                new_content = ""
                if content.get("audio", ""):
                    audio_binary = base64.b64decode(content["audio"])
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                        temp_file.write(audio_binary)
                        temp_file_path = temp_file.name
                        waveform = get_waveform(temp_file_path)
                        audios.append(waveform)
                    new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                elif content.get("text", ""):
                    new_content += content["text"]
            else:
                raise NotImplementedError
            new_messages.append({"role": f"{role}", "content": f"{new_content}"})

        prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompt += DEFAULT_TTS_START_TOKEN
        segments = prompt.split(f"{DEFAULT_AUDIO_TOKEN}")
        input_ids = []
        for idx, segment in enumerate(segments):
            if idx != 0:
                input_ids += [AUDIO_TOKEN_INDEX]
            input_ids += self.tokenizer.encode(segment)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)

        if audios:
            speech_inputs = self.audio_extractor(
                audios,
                sampling_rate=self.audio_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt"
            )
            speech_values = speech_inputs.input_features
            speech_mask = speech_inputs.attention_mask
        else:
            speech_values, speech_mask = None, None
        
        return input_ids, speech_values, speech_mask

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model
        generation_config = deepcopy(self.generation_config)
        tts_generation_config = deepcopy(self.tts_generation_config)

        messages = params["messages"]
        input_ids, speech_values, speech_mask = self.get_input_params(messages)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if speech_values is not None:
            speech_values = speech_values.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
            speech_mask = speech_mask.to(device='cuda', non_blocking=True)

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        do_sample = True if temperature > 0.001 else False

        generation_config.update(
            **{
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
        )
        tts_generation_config.update(
            **{
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0
            }
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True, timeout=15)
        units_streamer = TokenStreamer(skip_prompt=False, timeout=15)

        thread = Thread(target=model.generate, kwargs=dict(
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
                    yield json.dumps({"text": generated_text, "audio": "", "finalize": False, "error_code": 0}).encode() + b"\0"
                except StopIteration:
                    active_text = False
            if active_units:
                try:
                    new_unit = next(units_streamer)
                    units.append(new_unit - self.units_bias)
                    if len(units) >= block_size:
                        tts_token = torch.LongTensor(units).unsqueeze(0).to(device='cuda')
                        if prev_mel is not None:
                            prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                        tts_speech, tts_mel = self.audio_decoder.token2wav(tts_token, uuid=this_uuid,
                            prompt_token=flow_prompt_speech_token.to(device='cuda'),
                            prompt_feat=prompt_speech_feat.to(device='cuda'),
                            finalize=False)
                        prev_mel = tts_mel
                        tts_mels.append(tts_mel)
                        generated_audio = tts_speech.cpu()
                        buffer = BytesIO()
                        torchaudio.save(buffer, generated_audio, 22050, format="wav")
                        audio_binary = buffer.getvalue()
                        base64_string = base64.b64encode(audio_binary).decode("utf-8")
                        flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                        units = []
                        yield json.dumps({"text": generated_text, "audio": base64_string, "finalize": False, "error_code": 0}).encode() + b"\0"
                except StopIteration:
                    active_units = False
                    if units:
                        tts_token = torch.LongTensor(units).unsqueeze(0).to(device='cuda')
                        if prev_mel is not None:
                            prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                        tts_speech, tts_mel = self.audio_decoder.token2wav(tts_token, uuid=this_uuid,
                            prompt_token=flow_prompt_speech_token.to(device='cuda'),
                            prompt_feat=prompt_speech_feat.to(device='cuda') if prev_mel is not None else None,
                            finalize=True)
                        generated_audio = tts_speech.cpu()
                        buffer = BytesIO()
                        torchaudio.save(buffer, generated_audio, 22050, format="wav")
                        audio_binary = buffer.getvalue()
                        base64_string = base64.b64encode(audio_binary).decode("utf-8")
                        units = []
                        yield json.dumps({"text": generated_text, "audio": base64_string, "finalize": True, "error_code": 0}).encode() + b"\0"
                    else:
                        yield json.dumps({"text": generated_text, "audio": "", "finalize": True, "error_code": 0}).encode() + b"\0"


    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-name", type=str, default="omnispeech")
    parser.add_argument("--model-path", type=str, default="./OpenS2S")
    parser.add_argument("--flow-path", type=str, default="./glm-4-voice-decoder")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.flow_path,
                         args.model_name
                         )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
