import torch

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from accelerate import init_empty_weights, infer_auto_device_map
from qwen_omni_utils import process_mm_info
import soundfile as sf


class Qwen25OmniChat:
    def __init__(self, args):
        self.args = args
        with init_empty_weights():
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-Omni-7B", 
                torch_dtype="auto", 
            )
        device_map = infer_auto_device_map(self.model)

        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", 
            torch_dtype="auto", 
            device_map=device_map,
        )
        self.processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
        print(f"model loaded: {self.model.device} {self.model.dtype}")


    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt,
        user_instruction=None,
        audio_output=None,
        max_new_tokens=2048,
    ):

        USE_AUDIO_IN_VIDEO = True
        messages = []
        if len(system_prompt) > 0:
            messages.append(
                {
                    "role": "system", "content": [
                        {"type": "text", "text": system_prompt}
                    ] # "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                })
        if user_instruction:
            messages.append(
                {
                    "role": "user", "content": [
                        {"type": "text", "text": user_instruction},
                        {"type": "audio", "audio": audio},
                    ]
                }
            )
        else:
            messages.append(
                {
                    "role": "user", "content": [
                        {"type": "audio", "audio": audio},
                    ]
                }
            )
        
        try:
            
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False)
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            if audio_output:
                with torch.no_grad():
                    text_ids, audio_ids = self.model.generate(
                        **inputs,
                        return_audio=True,
                        max_new_tokens=max_new_tokens
                    )
            
                text_ids = text_ids[:,inputs['input_ids'].size(1):]

                text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                sf.write(
                    output_file,
                    audio_ids.reshape(-1).detach().cpu().numpy(),
                    samplerate=24000,
                )
                return {
                    "response_audio_transcript": text,
                    "response_text": "",
                }
            else:
                with torch.no_grad():
                    text_ids = self.model.generate(
                        **inputs,
                        return_audio=False,
                        max_new_tokens=max_new_tokens
                    )
                text_ids = text_ids[:,inputs['input_ids'].size(1):]
                text = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                return {
                    "response_audio_transcript": text,
                    "response_text": "",
                }
                
                
        except Exception as e:
            print("reason: ", e)
            print(f"error: {audio}")
            return {
                "response_audio_transcript": "",
                "response_text": "",
            }
        

