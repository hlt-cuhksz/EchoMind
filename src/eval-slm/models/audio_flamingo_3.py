import os

import llava
from llava import conversation as clib
from llava.media import Sound
from peft import PeftModel
import torch

from models.base import BaseS2SModel


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class AudioFlamingo3(BaseS2SModel):
    def __init__(self, args, chat=False, think_mode=False):
        super().__init__(args)
        self.args = args
        if chat:
            self.model_path = "/share/workspace/shared_models/01_LLM-models/audio-flamingo-3-chat"
        else:
            self.model_path = "/share/workspace/shared_models/01_LLM-models/audio-flamingo-3"

        self.model = llava.load(self.model_path)

        if think_mode:
            model_think = os.path.join(self.model_path, 'stage35')
            self.model = PeftModel.from_pretrained(
                self.model,
                model_think,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        
    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt=None,
        user_instruction=None,
        audio_output=False, # Ignored
        max_new_tokens=2048,
    ):
        generation_config = self.model.default_generation_config

        # Prepare multi-modal prompt
        prompt = []
        if audio is not None:
            if any(audio.endswith(ext) for ext in [".wav",".mp3", ".flac"]):
                audio = Sound(audio)
            else:
                raise ValueError(f"Unsupported audio type: {audio}")
            prompt.append(audio)
            
        if system_prompt and user_instruction:
            text_prompt = f"{system_prompt}\n{user_instruction}"
        elif system_prompt:
            text_prompt = f"{system_prompt}"
        elif user_instruction:
            text_prompt = f"{user_instruction}"
        else:
            text_prompt = f"<sound>"

        prompt.append(text_prompt)

        # Generate response
        response = self.model.generate_content(prompt, response_format=None, generation_config=generation_config)
        
        return {
            "response_audio_transcript": response,
            "response_text": response
        }

class AudioFlamingo3Chat(AudioFlamingo3):
    def __init__(self, args):
        super().__init__(args, chat=True)

class AudioFlamingo3Think(AudioFlamingo3):
    def __init__(self, args):
        super().__init__(args, think_mode=True)
