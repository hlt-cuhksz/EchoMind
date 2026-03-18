import soundfile as sf
from abc import ABC, abstractmethod
import sys
sys.path.append('echomind-master/src/eval-slm/models/Kimi-Audio')

from backoff import on_exception, expo
from kimia_infer.api.kimia import KimiAudio


class BaseS2SModel(ABC):
    def __init__(self, args):
        pass
    
    @abstractmethod
    @on_exception(expo, Exception, max_tries=3)
    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt,
        max_new_tokens
    ):
        pass


class KimiAudioS2SModel(BaseS2SModel):
    def __init__(self, args):
        super().__init__(args)
        self.model_path = getattr(args, 'model_path', "Kimi-Audio-7B-Instruct")#your local Kimi-Audio-7B-Instruct path
        self.sample_rate = getattr(args, 'sample_rate', 24000)
        self.model = KimiAudio(model_path=self.model_path, load_detokenizer=True)
    
    def generate_audio(self, audio, output_file, system_prompt, user_instruction, audio_output=None, max_new_tokens=1024):

        if system_prompt:
            if isinstance(system_prompt, (tuple, list)):
                if len(system_prompt) == 1:
                    clean_prompt = str(system_prompt[0])
                else:
                    clean_prompt = " ".join(str(item) for item in system_prompt)
            elif isinstance(system_prompt, str):
                clean_prompt = system_prompt.strip()
            else:
                clean_prompt = str(system_prompt)

            clean_prompt = clean_prompt + user_instruction
            print(clean_prompt)
            print(audio)
           
            messages_conversation = [
                {"role": "user", "message_type": "audio", "content": audio},
                {"role": "user", "message_type": "text", "content": clean_prompt}
            ]
        else:
            messages_conversation = [
                {"role": "user", "message_type": "audio", "content": audio}
            ]
        
        sampling_params = {"max_new_tokens": max_new_tokens}
        
        if audio_output:
            wav_output, text_output = self.model.generate(
                messages_conversation, 
                **sampling_params, 
                output_type="both"
            )
            
            sf.write(
                output_file, 
                wav_output.detach().cpu().view(-1).numpy(), 
                self.sample_rate
            )
            
            return {
                "response_text": None,
                "response_audio_transcript": text_output,
                "output_audio_path": output_file,
                "success": True
            }
        else:
            text_output = self.model.generate(
                messages_conversation, 
                **sampling_params, 
                output_type="text"
            )
            
            return {
                "response_text": text_output,
                "response_audio_transcript": text_output,               
                "output_audio_path": None,
                "success": True
            }
