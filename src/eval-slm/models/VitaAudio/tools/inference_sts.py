import json
import logging
import os
import random
import re
import sys
import time
import uuid
from threading import Thread
from typing import Optional

import torch
import tqdm
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.generation import GenerationConfig

import torchaudio
from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
from vita_audio.tokenizer import get_audio_tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch.manual_seed(1234)

# print visible devices
print(torch.cuda.device_count())

device_map = "cuda:0"
audio_tokenizer_rank = 0
torch_dtype = torch.bfloat16

# model_name_or_path = sys.argv[1]
# audio_tokenizer_path = sys.argv[2]
# flow_path = sys.argv[3]


class TextAudioIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        # self.audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
        self.audio_offset = tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")
        self.num_decode_tokens = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.num_decode_tokens += len(value)

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        elif self.token_cache[-1] >= self.audio_offset:
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

        while self.text_queue.qsize() > 10:
            time.sleep(0.01)


class BenchmarkIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, timeout, **decode_kwargs)

        self.num_decode_tokens = 0

    def put(self, value):
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.num_decode_tokens += len(value)

        printable_text = " ".join([str(x) for x in value.tolist()]) + " "
        self.on_finalized_text(printable_text)


def find_audio_segments_regex(text):
    """
    Find all substrings between <|begin_of_audio|> and <|end_of_audio|> using regex.

    Args:
        text (str): The input string to search through

    Returns:
        list: A list of all found audio segments (substrings between the delimiters)
    """
    pattern = re.compile(r"<\|begin_of_audio\|>(.*?)<\|end_of_audio\|>", re.DOTALL)
    segments = pattern.findall(text)
    return [segment.strip() for segment in segments]


def extract_token_ids_as_int(text):
    pattern = re.compile(r"<\|audio_(\d+)\|>")
    token_ids = pattern.findall(text)
    return [int(id) for id in token_ids]


def custom_init_weights(module):
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.constant_(module.bias, 0)
    elif isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(module.weight, 1)
        torch.nn.init.constant_(module.bias, 0)


class S2SInference:
    def __init__(
        self, model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path=None
    ):

        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

        if "qwen2" in config.model_type.lower():
            from evaluation.get_chat_template import qwen2_chat_template as chat_template

            add_generation_prompt = True

            default_system_message = []

        if "hunyuan" in config.model_type.lower():
            from evaluation.get_chat_template import hunyuan_chat_template as chat_template

            add_generation_prompt = False

            default_system_message = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant.",
                }
            ]

        luke_system_message = [
            {
                "role": "system",
                "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
            },
        ]

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )
        # print(f"{tokenizer=}")
        # print(f"{tokenizer.get_chat_template()=}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).eval()
        # print("model", model)
        # print(f"{model.config.model_type=}")
        # print(f"{model.hf_device_map=}")

        model.generation_config = GenerationConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        model.generation_config.max_new_tokens = 1024
        model.generation_config.chat_format = "chatml"
        model.generation_config.max_window_size = 8192
        # model.generation_config.use_cache = True
        model.generation_config.use_cache = False
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_k = 50
        model.generation_config.top_p = 1.0
        model.generation_config.num_beams = 1
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.config.model_type == "hunyuan":
            model.generation_config.eos_token_id = tokenizer.eos_id
        # print(f"{model.generation_config=}")

        audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
            flow_path=flow_path,
            rank=audio_tokenizer_rank,
        )

        self.model = model
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.add_generation_prompt = add_generation_prompt
        self.default_system_message = default_system_message
        self.luke_system_message = luke_system_message

        audio_0_id = tokenizer("<|audio_0|>").input_ids[0]
        # print(f"{audio_0_id=}")

    def benchmark_forward(self, mtp_inference_mode):
        # print("-" * 100)
        # print("benchmark_forward...")
        # print(f"{mtp_inference_mode=}")

        total_time = 0

        past_key_values = None
        use_cache = True

        self.model.input_ids = None
        self.model.inputs_embeds = None
        self.model.hidden_states = [None] * (self.model.config.num_nextn_predict_layers + 1)
        self.model.position_ids = None
        self.model.attention_mask = None
        self.model.mtp_idx = -1
        self.model.num_prefill_tokens = -1

        model_max_length = 1024
        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model._prepare_mtp_for_generation(mtp_inference_mode, model_max_length)

        else:
            self.model._prepare_mtp_for_generation(
                self.model.generation_config.mtp_inference_mode, model_max_length
            )

        for i in tqdm.tqdm(range(1, model_max_length + 1)):

            if use_cache:
                input_ids = torch.tensor([i - 1], dtype=torch.long).unsqueeze(0).to("cuda")
                position_ids = torch.tensor([i - 1], dtype=torch.long).unsqueeze(0).to("cuda")
            else:
                input_ids = torch.arange(i, dtype=torch.long).unsqueeze(0).to("cuda")
                position_ids = torch.arange(i, dtype=torch.long).unsqueeze(0).to("cuda")

            attention_mask = torch.tensor([1] * i, dtype=torch.float).unsqueeze(0).to("cuda")

            torch.cuda.synchronize()
            start = time.time()

            output = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                num_logits_to_keep=1,
            )

            torch.cuda.synchronize()
            end = time.time()

            total_time += end - start
            # print(f"{i=} {total_time=}")

            past_key_values = output.past_key_values

        # print()
        # print(f"{total_time=}")
        # print(f"second/token {total_time/model_max_length=}")
        # print(f"token/second {model_max_length/total_time=}")

        if mtp_inference_mode is not None:
            self.model.mtp_inference_mode = ori_mtp_inference_mode

    def benchmark_generate(self, mtp_inference_mode):

        self.model.apply(custom_init_weights)

        # print("-" * 100)
        # print("benchmark_generate...")
        # print(f"{mtp_inference_mode=}")

        total_time = 0
        self.model.generation_config.use_cache = True

        self.model.generation_config.max_new_tokens = 8192

        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        input_ids = torch.tensor([0], dtype=torch.long).unsqueeze(0).to("cuda")

        torch.cuda.synchronize()
        start = time.time()

        output = self.model.generate(
            input_ids,
        )
        # print(f"{output.size()=}")

        torch.cuda.synchronize()
        end = time.time()

        total_time += end - start

        # print()
        # print(f"{total_time=}")
        # print(f"second/token {total_time/output.size(1)=}")
        # print(f"token/second {output.size(1)/total_time=}")

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode

    def benchmark_generate_stream(self, mtp_inference_mode):
        # print("-" * 100)
        # print("benchmark_generate_stream...")
        # print(f"{mtp_inference_mode=}")

        self.model.apply(custom_init_weights)

        total_time = 0
        self.model.generation_config.use_cache = True

        # model_max_length = 8192
        model_max_length = 4096
        # model_max_length = 2048
        # model_max_length = 1024
        num_prefill_tokens = 32

        self.model.generation_config.max_new_tokens = model_max_length
        self.model.generation_config.do_sample = False

        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        input_ids = torch.tensor([0] * num_prefill_tokens, dtype=torch.long).unsqueeze(0).to("cuda")

        streamer = BenchmarkIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(input_ids=input_ids, streamer=streamer)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        token_decode_time = []

        torch.cuda.synchronize()
        start = time.time()
        thread.start()

        generated_text = ""
        for new_text in tqdm.tqdm(streamer, total=model_max_length):
            generated_text += new_text
            end = time.time()

            token_decode_time.append(end - start)

            yield new_text

        # print(f"{len(generated_text)}")

        torch.cuda.synchronize()
        end = time.time()

        total_time += end - start

        # print()
        # print(f"{token_decode_time[-1]=}")
        # print(f"{streamer.num_decode_tokens=}")
        # print(f"second/token {token_decode_time[-1]/streamer.num_decode_tokens=}")
        # print(f"token/second {streamer.num_decode_tokens/token_decode_time[-1]=}")

        # if mtp_inference_mode is None:
        #     mtp_inference_mode = []
        # with open(f'token_decode_time_{str(mtp_inference_mode)}.json', 'w') as f:
        #     json.dump(token_decode_time, f)

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode

    def run_infer(
        self,
        audio_path=None,
        prompt_audio_path=None,
        stream_stride=4,
        max_returned_tokens=4096,
        sample_rate=16000,
        request_id="",
        audio_feats=None,
        message="",
        use_past=False,
        mode="luke",
        do_sample=False,
        mtp_inference_mode=None,
    ):

        AUD_TAG_TOKEN = "<|audio|>"
        AUD_CONTEXT_TOKEN = "<|context_of_audio|>"
        AUD_START_TOKEN = "<|begin_of_audio|>"
        AUD_END_TOKEN = "<|end_of_audio|>"

        if prompt_audio_path is not None:
            system_message = [
                {
                    "role": "system",
                    "content": f"Your Voice: <|audio|>\n",
                },
            ]

        elif mode == "luke":
            system_message = self.luke_system_message

        else:
            system_message = self.default_system_message

        if prompt_audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(prompt_audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            system_message[-1]["content"] = system_message[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        if audio_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        else:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        if (audio_path is not None or prompt_audio_path is not None) and self.audio_tokenizer.apply_to_role(
            "user", is_contiguous=True
        ):
            # contiguous codec
            audio_paths = []
            if audio_path is not None:
                audio_paths.append(audio_path)
            if prompt_audio_path is not None:
                audio_paths.append(prompt_audio_path)
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")

        # print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

        self.model.generation_config.do_sample = do_sample

        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        outputs = self.model.generate(
            input_ids,
            audios=audios,
            audio_indices=audio_indices,
        )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # print(f"{output=}", flush=True)

        audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")

        audio_tokens = []
        for token_id in outputs[0]:
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)

        if len(audio_tokens) > 0:
            tts_speech = self.audio_tokenizer.decode(
                audio_tokens, source_speech_16k=prompt_audio_path
            )

        else:
            tts_speech = None

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode

        return output, tts_speech

    def run_infer_stream(
        self,
        audio_path=None,
        prompt_audio_path=None,
        stream_stride=4,
        max_returned_tokens=4096,
        sample_rate=16000,
        request_id="",
        audio_feats=None,
        message="",
        use_past=False,
        mode="luke",
        do_sample=False,
        mtp_inference_mode=None,
    ):

        if prompt_audio_path is not None:
            system_message = [
                {
                    "role": "system",
                    "content": f"Your Voice: <|audio|>\n",
                },
            ]

        elif mode == "luke":
            system_message = self.luke_system_message

        else:
            system_message = self.default_system_message

        if prompt_audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(prompt_audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            system_message[-1]["content"] = system_message[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        if audio_path is not None:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message + "\n<|audio|>",
                },
            ]
        else:
            messages = system_message + [
                {
                    "role": "user",
                    "content": message,
                },
            ]

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        if (audio_path is not None or prompt_audio_path is not None) and self.audio_tokenizer.apply_to_role(
            "user", is_contiguous=True
        ):
            # contiguous codec
            audio_paths = []
            if audio_path is not None:
                audio_paths.append(audio_path)
            if prompt_audio_path is not None:
                audio_paths.append(prompt_audio_path)
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, audio_paths, self.tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")

        # print("input", self.tokenizer.decode(input_ids[0], skip_special_tokens=False), flush=True)

        self.model.generation_config.do_sample = do_sample

        if mtp_inference_mode is not None:
            ori_mtp_inference_mode = self.model.generation_config.mtp_inference_mode
            self.model.generation_config.mtp_inference_mode = mtp_inference_mode

        streamer = TextAudioIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(
            input_ids=input_ids,
            audios=audios,
            audio_indices=audio_indices,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)

        thread.start()

        # generated_text = ""
        for new_text in streamer:
            # generated_text += new_text

            yield new_text

        # torch.cuda.synchronize()

        if mtp_inference_mode is not None:
            self.model.generation_config.mtp_inference_mode = ori_mtp_inference_mode


def benchmark_llm():

    for mtp_inference_mode, tag in zip(
        [
            [8192, 0],
            [1, 4, 3, 8, 4, 10],
            [1, 10, 4, 10],
            [1, 10],
        ],
        [
            "Vanilla",
            "Balance",
            "Boost",
            "Turbo",
        ],
    ):
        # print("=" * 100)
        # print("benchmark_llm")
        # print(f"{tag}")

        s2s_inference.benchmark_forward(mtp_inference_mode)

        s2s_inference.benchmark_generate(mtp_inference_mode)

        generated_text = ""
        for new_text in s2s_inference.benchmark_generate_stream(
            mtp_inference_mode=mtp_inference_mode
        ):
            generated_text += new_text
            # print(new_text, end="", flush=True)


def benchmark_sts():
    audio_paths = [
        "asset/介绍一下上海.wav",
        "asset/发表一个悲伤的演讲.wav",
        "asset/发表一个振奋人心的演讲.wav",
    ]

    for _ in range(10):

        # print("=" * 100)
        # print("benchmark_sts")
        # audio_path = random.choice(audio_paths)
        # print(f"{audio_path}")

        start = time.time()
        audio_idx = 0
        generated_text = ""
        all_tts_speech = []
        past_tts_speech_len = 0
        for new_text in s2s_inference.run_infer_stream(audio_path=audio_path):
            # print(new_text, end="", flush=True)

            generated_text += new_text

            if new_text == "<|end_of_audio|>":
                audio_tokens = extract_token_ids_as_int(generated_text)

                tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens, option_steps=1)
                tts_speech = tts_speech[past_tts_speech_len:]
                past_tts_speech_len += len(tts_speech)
                all_tts_speech.append(tts_speech)

                end = time.time()
                # if audio_idx == 0:
                #     print(audio_tokens)
                # print(f"{audio_idx} audio chunk {end - start}")

                wav_path = os.path.join(output_dir, audio_path[:-4] + f"_{audio_idx}.wav")
                os.makedirs(os.path.dirname(wav_path), exist_ok=True)
                torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

                audio_idx += 1
                start = time.time()

        wav_path = os.path.join(output_dir, audio_path[:-4] + ".wav")
        tts_speech = torch.cat(all_tts_speech, dim=0)
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# Text
def text_task():
    for text in [
        "How many helicopters can a human eat in one sitting?",
        "你叫什么名字？",
        "写一首诗",
        "介绍一下上海",
    ]:
        # print("=" * 100)
        # print("text_task")
        # print(f"{text=}")

        output, _ = s2s_inference.run_infer(
            message=text,
            mode=None,
            # do_sample=True,
            mtp_inference_mode=[8192, 0],
        )
        # print(f"{output=}", flush=True)


# ==============================================================
# Text stream
def text_stream_task():
    for text in [
        "你叫什么名字？",
    ]:
        print("=" * 100)
        print("text_stream_task")
        print(f"{text=}")

        generated_text = ""
        for new_text in s2s_inference.run_infer_stream(
            message=text,
            mode=None,
            # do_sample=True,
            mtp_inference_mode=[8192, 0],
        ):
            generated_text += new_text
            print(new_text, end="")
        print("")


# ==============================================================
# S2S
def sts_task():
    for audio_path in [
        "asset/介绍一下上海.wav",
        "asset/发表一个悲伤的演讲.wav",
        "asset/发表一个振奋人心的演讲.wav",
        "asset/piano.mp3",
    ]:
        print("=" * 100)
        print("sts_task")
        print(f"{audio_path=}")

        output, tts_speech = s2s_inference.run_infer(
            audio_path=audio_path,
        )

        wav_path = os.path.join(output_dir, audio_path[:-4] + ".wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# S2S stream
def sts_stream_task():
    for audio_path in [
        "asset/介绍一下上海.wav",
    ]:
        print("=" * 100)
        print("sts_stream_task")
        print(f"{audio_path=}")

        generated_text = ""
        for new_text in s2s_inference.run_infer_stream(audio_path=audio_path):
            generated_text += new_text
            print(new_text, end="")
        print("")

        audio_decode_time = []
        audio_segments = find_audio_segments_regex(generated_text)
        for audio_idx, audio_segment in enumerate(audio_segments):
            start = time.time()

            audio_tokens = extract_token_ids_as_int(audio_segment)
            # print(audio_tokens)

            tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens)

            end = time.time()
            audio_decode_time.append(end - start)

            wav_path = os.path.join(output_dir, audio_path[:-4] + f"_{audio_idx}.wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")
        # print(f"{audio_decode_time=}")


# ==============================================================
# ASR
def asr_task():
    for audio_path in [
        "/data/data/wenet-e2e/wenetspeech/data/cuts_TEST_NET.00000000/TES/TEST_NET_Y0000000020_5XD21BihDd8_S00395.wav",
        "/data/data/wenet-e2e/wenetspeech/data/cuts_TEST_NET.00000000/TES/TEST_NET_Y0000000000_-KTKHdZ2fb8_S00424.wav",
        "/data/data/wenet-e2e/wenetspeech/data/cuts_TEST_NET.00000000/TES/TEST_NET_Y0000000050_LOLTeK1BNMo_S00045.wav",
        "/data/data/fixie-ai/librispeech_asr/test.clean/2830-3980-0034.wav",
        "/data/data/fixie-ai/librispeech_asr/test.clean/237-134500-0040.wav",
    ]:
        print("=" * 100)
        print("asr_task")
        print(f"{audio_path=}")

        output, tts_speech = s2s_inference.run_infer(
            audio_path=audio_path,
            # message="Translate the speech to text.",
            message="Convert the speech to text.",
            mode=None,
        )
        print(f"{output=}", flush=True)


# ==============================================================
# TTS
def tts_task():
    TTS_texts = [
        "我们将为全球城市的可持续发展贡献力量。",
        "通天河 灵感大王",
        "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
        "一二三四五六七八九十",
        "One Two Tree Four Five Six Seven Eight Night Ten",
        "1 2 3 4 5 6 7 8 9 10",
        "12345678910",
        "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
        "坡上立着一只鹅，坡下就是一条河。宽宽的河，肥肥的鹅，鹅要过河，河要渡鹅不知是鹅过河，还是河渡鹅?",
        "扁担长，板凳宽，扁担没有板凳宽，板凳没有扁担长。扁担绑在板凳上，板凳不让扁担绑在板凳上。",
        "化肥会挥发，黑化肥发灰，灰化肥发黑。黑化肥发灰会挥发；灰化肥挥发会发黑。黑化肥挥发发灰会花飞；灰化肥挥发发黑会飞花，黑灰化肥会挥发发灰黑讳为花飞；灰黑化肥会挥发发黑灰为讳飞花。",
        "圆桌儿、方桌儿没有腿儿，墨水瓶儿里没有水儿，花瓶里有花儿没有叶儿，练习本儿上写字儿没有准儿，甘蔗好吃净是节儿。西瓜挺大没有味儿，坛儿里的小米儿长了虫儿，鸡毛掸子成了棍儿，水缸沿儿上系围裙儿，耗子打更猫打盹儿，新买的小褂儿没钉扣儿，奶奶想说没有劲儿。",
        "起床歌：小宝宝，起得早，睁开眼，眯眯笑，咿呀呀，学说话，伸伸手，要人抱。穿衣歌小胳膊，穿袖子，穿上衣，扣扣子，小脚丫，穿裤子，穿上袜子穿鞋子。小镜子-小镜子，圆又圆，看宝宝，露笑脸。闭上眼，做个梦，变月亮，挂上天。小铃铛叮铃铃，叮铃铃，一会远，一会近。小宝宝，耳朵灵，听铃声，找到铃。学画画小宝宝，学画画，大蜡笔，手中拿，画小鸭，叫嘎嘎，画小马，骑回家。大鞋子大鞋子，像只船，爸爸穿，我也穿，一二一，向前走，走呀走，翻了船。逛公园逛公园，宝宝笑，东看看，西瞧瞧，花儿香，鸟儿叫，小草绿，小树摇。看画报小娃娃，看画报，睁大眼，仔细瞧，布娃娃，哈哈笑，伸伸手，要你抱。搭积木大积木，红黄兰，小宝宝，最爱玩，搭火车，钻山洞，盖高楼，连着天。小汽车小汽车，嘀嘀嘀，开过来，开过去，小宝宝，当司机，送妈妈，上班去。藏猫猫儿歌：躲猫猫，躲猫猫， 猫猫、猫猫在哪里？喵……猫咪在这里。",
    ]

    for text in TTS_texts:
        print("=" * 100)
        print("tts_task")
        print(f"{text=}")

        output, tts_speech = s2s_inference.run_infer(
            message="Convert the text to speech.\n" + text,
            mode=None,
            do_sample=True,
        )

        wav_path = os.path.join(output_dir, text[:16] + ".wav")
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

    # ==============================================================
    # Clone TTS
    for text in TTS_texts:
        for prompt_audio_path in [
            "asset/2631296891109983590.wav",
            "asset/379838640-d5ff0815-74f8-4738-b0f1-477cfc8dcc2d.wav",
            "asset/4202818730519913143.wav",
        ]:
            print("=" * 100)
            print("tts_task")
            print(f"{text=} {prompt_audio_path=}")

            output, tts_speech = s2s_inference.run_infer(
                prompt_audio_path=prompt_audio_path,
                # message="Translate the text to speech.\n" + text,
                message="Convert the text to speech.\n" + text,
                mode=None,
                do_sample=True,
            )

            wav_path = os.path.join(output_dir, prompt_audio_path[:16] + "_" + text[:16] + ".wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


# ==============================================================
# TTS stream
def tts_stream_task():
    TTS_texts = [
        "他本是我莲花池里养大的金鱼，每日浮头听经，修成手段。那一柄九瓣铜锤，乃是一枝未开的菡萏，被他运炼成兵。不知是那一日，海潮泛涨，走到此间。我今早扶栏看花，却不见这厮出拜，掐指巡纹，算着他在此成精，害你师父，故此未及梳妆，运神功，织个竹篮儿擒他。",
    ]

    for text in TTS_texts:
        print("=" * 100)
        print("tts_stream_task")
        print(f"{text=}")

        generated_text = ""
        for new_text in s2s_inference.run_infer_stream(
            message="Convert the text to speech.\n" + text,
            mode=None,
            do_sample=True,
        ):
            generated_text += new_text
            print(new_text, end="")
        print("")

        audio_segments = find_audio_segments_regex(generated_text)
        for audio_idx, audio_segment in enumerate(audio_segments):
            audio_tokens = extract_token_ids_as_int(audio_segment)
            # print(audio_tokens)
            tts_speech = s2s_inference.audio_tokenizer.decode(audio_tokens)

            wav_path = os.path.join(output_dir, text[:16] + f"_{audio_idx}.wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")


if __name__ == "__main__":
    s2s_inference = S2SInference(
        model_name_or_path, audio_tokenizer_path, audio_tokenizer_type, flow_path=flow_path
    )


    text_task()
    text_stream_task()

    sts_task()
    sts_stream_task()

    asr_task()
    tts_task()
    tts_stream_task()

    benchmark_sts()
    benchmark_llm()
