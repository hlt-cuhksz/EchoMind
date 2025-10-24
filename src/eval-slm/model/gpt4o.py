
from openai import OpenAI
# import soundfile as sf
import base64
import argparse
import torch
import json
import os
import time

class GPT4oChat:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(api_key=args.api_key)
        self.model_name = "gpt-4o-audio-preview"

    def encode_audio_to_base64(self, audio_file_path):
        with open(audio_file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
        
    def save_audio_from_base64(self, base64_audio, output_path):
        audio_data = base64.b64decode(base64_audio)
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)

    def generate_audio(
        self,
        audio,
        output_file,
        system_prompt,
        user_instruction=None,
        audio_output=None,
        max_new_tokens=2048,
    ):
        input_audio_base64 = self.encode_audio_to_base64(audio)
        if system_prompt:
            messages=[{"role": "system", "content": system_prompt}]
        else:
            messages=[]
        if user_instruction:
            messages.append(
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": user_instruction
                        },
                        {
                            "type": "input_audio", 
                            "input_audio": {
                                "data": input_audio_base64, 
                                "format": 'wav'
                                }
                        }]
                }
            )
        else:
            messages.append(
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "input_audio", 
                            "input_audio": {
                                "data": input_audio_base64, 
                                "format": 'wav'
                                }
                        }]
                },

            )


        try:
            if audio_output:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    modalities=["text", "audio"],
                    audio={"voice": "alloy", "format": "wav"},
                    max_tokens=max_new_tokens,
                    messages=messages
                )

                response = completion.choices[0].message
                time.sleep(2)

                if hasattr(response, 'audio') and response.audio:
                    self.save_audio_from_base64(response.audio.data, output_file)
                return {
                    "response_audio_transcript": response.audio.transcript,  # speech transcript
                    "response_text": response.content, # text response
                }
            else:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    max_tokens=max_new_tokens,
                    messages=messages
                )
                response = completion.choices[0].message
                return {
                    "response_audio_transcript": response.contentresponse.content,  # speech transcript
                    "response_text": response.content, # text response
                }


        except Exception as e:
            print("reason: ", e)
            print(f"error: {audio}")
            return {
                "response_audio_transcript": "",
                "response_text": "",
            }
