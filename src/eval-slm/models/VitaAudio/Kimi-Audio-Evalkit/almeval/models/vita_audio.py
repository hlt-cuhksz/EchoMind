from io import BytesIO
import sys

import librosa
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from ..utils.misc import print_once
from .base import BaseModel

from vita_audio.data.processor.audio_processor import add_audio_input_contiguous
from vita_audio.tokenizer import get_audio_tokenizer

chat_template = """
{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n
"""

class VITAAudio(BaseModel):
    NAME = 'VITA-Audio'

    def __init__(self,
                 model_path="VITA-MLLM/VITA-Audio-Plus-Boost",
                 device='cuda',
                 torch_dtype=torch.bfloat16,
                 **kwargs):
        self.device = device

        self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        self.vita_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        ).to(device).eval()

        self.vita_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            chat_template=chat_template,
        )

        self.vita_model.generation_config = GenerationConfig.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.vita_model.generation_config.max_new_tokens = 2048
        self.vita_model.generation_config.chat_format = "chatml"
        self.vita_model.generation_config.max_window_size = 2048
        self.vita_model.generation_config.use_cache = True
        # self.vita_model.generation_config.use_cache = False
        self.vita_model.generation_config.do_sample = False

        sys.path.append("glm4voice/")
        sys.path.append("glm4voice/cosyvoice/")
        sys.path.append("glm4voice/third_party/Matcha-TTS/")
        audio_tokenizer_path = "/data/models/THUDM/glm-4-voice-tokenizer"
        flow_path = "/data/models/THUDM/glm-4-voice-decoder"
        audio_tokenizer_type = "sensevoice_glm4voice"
        self.audio_tokenizer = get_audio_tokenizer(
            audio_tokenizer_path,
            audio_tokenizer_type,
            flow_path=flow_path,
            # rank=audio_tokenizer_rank,
        )

        self.default_system_message = [
        ]

        self.luke_system_message = [
            {
                "role": "system",
                "content": "Your Name: Luke\nYour Gender: male\n\nRespond in a text-audio interleaved manner.",
            },
        ]

        self.add_generation_prompt = True

        torch.cuda.empty_cache()

    def get_system_message(self, msg: dict):
        meta = msg['meta']
        if meta is None:
            return self.default_system_message
        if meta['task'] == 'ASR':
            return self.default_system_message

        return self.luke_system_message

    def get_task_message(self, msg: dict):
        meta = msg['meta']
        if meta['task'] == 'ASR':
            messages = [
                {
                    "role": "user",
                    "content": "Convert the speech to text.\n<|audio|>",
                },
            ]

        elif meta['interactive'] == 'Audio-QA':
            messages = [
                {
                    "role": "user",
                    "content":  "<|audio|>",
                },
            ]

        elif meta['audio_type'] == 'AudioEvent':
            messages = [
                {
                    "role": "user",
                    "content":  msg['text'] + "\n<|audio|>",
                },
            ]

        else:
            messages = [
                {
                    "role": "user",
                    "content":  msg['text'] + "\n<|audio|>",
                },
            ]

        return messages


    def generate_inner(self, msg: dict):
        audio_path = msg['audio']
        if len(audio_path) == 1:
            audio_path = audio_path[0]

        prompt_audio_path = None
        messages = self.get_task_message(msg)
        system_message = self.get_system_message(msg)

        # only for dump
        messages = system_message + messages
        print_once(f'messages: {messages}')

        if prompt_audio_path is not None:
            if self.audio_tokenizer.apply_to_role("system", is_discrete=True):
                # discrete codec
                prompt_audio_tokens = self.audio_tokenizer.encode(prompt_audio_path)
                prompt_audio_tokens = "".join(f"<|audio_{i}|>" for i in prompt_audio_tokens)
                system_message = [
                    {
                        "role": "system",
                        "content": f"Your Voice: <|begin_of_audio|>{prompt_audio_tokens}<|end_of_audio|>\n",
                    },
                ]

            else:
                # contiguous codec
                system_message = system_message

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_discrete=True):
            # discrete codec
            audio_tokens = self.audio_tokenizer.encode(audio_path)
            audio_tokens = "".join(f"<|audio_{i}|>" for i in audio_tokens)
            messages[-1]["content"] = messages[-1]["content"].replace(
                "<|audio|>", f"<|begin_of_audio|>{audio_tokens}<|end_of_audio|>"
            )

        input_ids = self.vita_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )

        if audio_path is not None and self.audio_tokenizer.apply_to_role("user", is_contiguous=True):
            # contiguous codec
            input_ids, audios, audio_indices = add_audio_input_contiguous(
                input_ids, [audio_path], self.vita_tokenizer, self.audio_tokenizer
            )
        else:
            audios = None
            audio_indices = None

        input_ids = torch.tensor([input_ids], dtype=torch.long).to("cuda")

        responses = self.vita_model.generate(
            input_ids,
            audios=audios,
            audio_indices=audio_indices,
        )

        response = responses[0][len(input_ids[0]) :]

        # audio_offset = self.vita_tokenizer.convert_tokens_to_ids("<|audio_0|>")
        audio_offset = self.vita_tokenizer.convert_tokens_to_ids("<|begin_of_audio|>")

        audio_tokens = []
        text_tokens = []
        for token_id in response:
            if token_id >= audio_offset:
                audio_tokens.append(token_id - audio_offset)
            else:
                text_tokens.append(token_id)

        # if len(audio_tokens) > 0:
        #     tts_speech = self.audio_tokenizer.decode(
        #         audio_tokens, source_speech_16k=prompt_audio_path
        #     )

        # else:
        #     tts_speech = None

        out_text = self.vita_tokenizer.decode(
            text_tokens, skip_special_tokens=True,
        )
        # print_once(f'{out_text=}')

        return self.vita_tokenizer.decode(input_ids[0], skip_special_tokens=False), out_text
