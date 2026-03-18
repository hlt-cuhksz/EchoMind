import os
import torch
import torchaudio
import re
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
from web_demo.generation import decode_wave_vocoder, GenerationAudioTokens
import time
import ujson
from web_demo.constants import *

# sys.path.append(os.path.join(COSY_VOCODER))
from third_party.cosy24k_vocoder.cosy24k_vocoder import Cosy24kVocoder


class AudioTextGenerator:
    def __init__(self, model_path, vocoder_path, sampling_rate=16000):
        self.model_path = model_path
        self.vocoder_path = vocoder_path
        self.sampling_rate = sampling_rate

        # Initialize the model and tokenizer
        self.model, self.tokenizer = self.init_model()

        # Initialize vocoder
        self.vocoder = Cosy24kVocoder.from_pretrained(self.vocoder_path).cuda()

        # Get special tokens
        self.audio_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_start_token_id
        )
        self.audio_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audio_end_token_id
        )
        self.audiogen_start_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audiogen_start_token_id
        )
        self.audiogen_end_token = self.tokenizer.convert_ids_to_tokens(
            self.model.config.audio_config.audiogen_end_token_id
        )
        self.special_token_pattern = re.compile(
            '<\|endoftext\|>|<audiogen_start_baichuan>|<audiogen_end_baichuan>'
        )

    def init_model(self):
        """Initialize the model and tokenizer"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        model.training = False
        model.bind_processor(tokenizer, training=False, relative_path="/")
        return model, tokenizer

    def generate_text(self, pret, plen):
        """Generate text response from the model"""
        textret = self.model.generate(
            pret.input_ids.cuda(),
            attention_mask=pret.attention_mask.cuda(),
            tokenizer=self.tokenizer,  # 显式传递 tokenizer
            max_new_tokens=1024,
            stop_strings=['<|endoftext|>'],
            do_sample=True, temperature=0.8, top_k=20, top_p=0.85, repetition_penalty=1.1,
            return_dict_in_generate=True
        )
        return textret, self.tokenizer.decode(textret.sequences[0, plen:])

    def generate_audio(self, pret):
        """Generate audio from the model"""
        audioret = GenerationAudioTokens.generate(
            self.model,
            pret.sequences,
            attention_mask=torch.ones_like(pret.sequences),
            max_new_tokens=500,
            do_sample=True, temperature=0.5, top_k=5, top_p=0.85, repetition_penalty=1.3,
            return_dict_in_generate=True
        )
        wave_segment = decode_wave_vocoder(audioret.audios_sequences.clone(), self.vocoder, self.model)
        return audioret, wave_segment

    def save_audio(self, wave, output_path):
        """Save generated audio to a file"""
        torchaudio.save(output_path, torch.cat(wave, dim=0).cpu(), self.sampling_rate)

    def process_audio(self, audio_path):
        """Load and process audio file"""
        wave, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            wave = torchaudio.functional.resample(wave, sr, self.sampling_rate)
        return wave

    def generate_response(self, audio_path, system_prompt, max_new_tokens=1024):
        """Generate both text and audio response from the model"""
        # Process the input audio
        input_audio_path = os.path.join(g_cache_dir, "input.wav")
        os.system(f"cp {audio_path} {input_audio_path}")

        # Prepare messages
        pret = self.model.processor([system_prompt])
        plen = pret.input_ids.shape[1]

        # Generate text
        ret, text_segment = self.generate_text(pret, plen)
        full_text = re.sub(self.special_token_pattern, '', text_segment)
        show_text = re.sub(self.special_token_pattern, '', text_segment)

        # Generate audio
        wave_list = []
        ret.sequences[0, -1] = self.model.config.audio_config.audiogen_start_token_id
        ret, wave_segment = self.generate_audio(ret)
        wave_list.extend(wave_segment)

        # Save generated audio
        audio_output_path = os.path.join(g_cache_dir, 'output_audio.wav')
        self.save_audio(wave_list, audio_output_path)

        # Return generated text and audio
        return full_text, audio_output_path


# Example usage:
audio_text_generator = AudioTextGenerator(MODEL_PATH, COSY_VOCODER)

# Assuming audio_path is the path to the input audio file and system_prompt is your system prompt
audio_path = "/share/home/lutong/Baichuan-Omni-1.5/web_demo/data/1497.wav"
system_prompt = "这个说话者是男是女？"

generated_text, generated_audio = audio_text_generator.generate_response(audio_path, system_prompt)

print(f"Generated Text: {generated_text}")
print(f"Generated Audio saved at: {generated_audio}")
