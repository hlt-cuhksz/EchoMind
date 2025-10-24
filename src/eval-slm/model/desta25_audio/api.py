from desta import DeSTA25AudioModel
from models.base import BaseS2SModel


class DeSTA25Audio(BaseS2SModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # Load the model from Hugging Face
        self.model = DeSTA25AudioModel.from_pretrained("/share/workspace/shared_models/01_LLM-models/DeSTA2.5-Audio-Llama-3.1-8B")
        print("Model loaded successfully.")
        self.model.to("cuda")
        print("Model moved to GPU.")
        
    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt,
        user_instruction="",
        audio_output=False, # Ignored
        max_new_tokens=2048,
    ):
        # Run inference with audio input
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": f"<|AUDIO|>\n{user_instruction}" if user_instruction else "<|AUDIO|>",
            "audios": [{
                "audio": audio,
                "text": None
            }]
        })

        outputs = self.model.generate(
            messages=messages,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            max_new_tokens=max_new_tokens
        )
        
        return {
            "response_audio_transcript": outputs.text,
            "response_text": outputs.text
        }
