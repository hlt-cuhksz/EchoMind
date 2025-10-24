import sys
import os
import shutil
from abc import ABC, abstractmethod

sys.path.append('echomind-master/src/eval-slm/models/Baichuan-Omni-1.5/web_demo')
from oneaudio import AudioChatSystem


class BaseS2SModel(ABC):
    def __init__(self, args):
        pass
        
    @abstractmethod
    def generate_audio(self, audio, output_file, system_prompt, max_new_tokens=512):
        pass


class BaichuanChat(BaseS2SModel):
    def __init__(self, args):
        super().__init__(args)
        self.audio_chat_system = AudioChatSystem(
            model_path=getattr(args, 'model_path', None),
            vocoder_path=getattr(args, 'vocoder_path', None)
        )
    
    def generate_audio(self, audio, output_file, system_prompt, max_new_tokens=512):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.audio_chat_system.clear_history()
                
                if isinstance(system_prompt, (tuple, list)):
                    system_prompt_str = ' '.join(str(item) for item in system_prompt)
                else:
                    system_prompt_str = str(system_prompt)
                
                text_response, audio_response_path = self.audio_chat_system.generate_response(
                    audio_path=audio,
                    system_prompt=system_prompt_str
                )
                
                final_audio_path = audio_response_path
                if output_file and audio_response_path:
                    output_dir = os.path.dirname(output_file)
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                    shutil.copy2(audio_response_path, output_file)
                    final_audio_path = output_file
                
                return {
                    'response_audio_transcript': text_response,
                    'response_audio_path': final_audio_path
                }
                
            except Exception as e:
                if attempt == max_retries - 1:  
                    raise e
                print(f"try {attempt + 1} faile: {e}, retrying...")



