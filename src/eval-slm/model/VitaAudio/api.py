import os
import re

import torchaudio
from backoff import on_exception, expo

from .tools.inference_sts import S2SInference
from models.base import BaseS2SModel

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class VitaAudio(BaseS2SModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
        # Initialize S2S inference
        self.s2s_inference = S2SInference(
            model_name_or_path="/share/workspace/shared_models/01_LLM-models/VITA-Audio-Plus-Vanilla", 
            audio_tokenizer_path="/share/workspace/shared_models/01_LLM-models/glm-4-voice-tokenizer", 
            audio_tokenizer_type="sensevoice_glm4voice", 
            flow_path="/share/workspace/shared_models/01_LLM-models/glm-4-voice-decoder"
        )
    
    @on_exception(expo, Exception, max_tries=3)
    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt=None,
        user_instruction=None,
        audio_output=False,
        max_new_tokens=2048,
    ):
        if system_prompt and user_instruction:
            message = system_prompt + "\n" + user_instruction
        elif system_prompt:
            message = system_prompt
        elif user_instruction:
            message = user_instruction
        else:
            message = ""
        
        output, tts_speech = self.s2s_inference.run_infer(
            audio_path=audio,
            message=message,
            mode=None if not audio_output else "luke",
            max_returned_tokens=max_new_tokens,
        )

        if audio_output:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            torchaudio.save(output_file, tts_speech.unsqueeze(0), 22050, format="wav")

        response_text = re.sub(r"<\|.*?\|>", "", output.strip())

        return {
            "response_audio_transcript": response_text,
            "response_text": response_text
        }
