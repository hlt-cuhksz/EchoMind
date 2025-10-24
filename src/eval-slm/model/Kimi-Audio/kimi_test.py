import soundfile as sf
from kimia_infer.api.kimia import KimiAudio

# --- 1. Load Model ---
# model_path = "moonshotai/Kimi-Audio-7B-Instruct" 
model_path = "/online1/gzs_data/LLM-models/Kimi-Audio-7B-Instruct" 
model = KimiAudio(model_path=model_path, load_detokenizer=True)

# --- 2. Define Sampling Parameters ---
sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

# --- 3. Example 1: Audio-to-Text (ASR) ---
messages_asr = [
    # You can provide context or instructions as text
    {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
    # Provide the audio file path
    {"role": "user", "message_type": "audio", "content": "test_audios/asr_example.wav"}
]

# Generate only text output
_, text_output = model.generate(messages_asr, **sampling_params, output_type="text")
print(">>> ASR Output Text: ", text_output) # Expected output: "这并不是告别，这是一个篇章的结束，也是新篇章的开始。"


# --- 4. Example 2: Audio-to-Audio/Text Conversation ---
messages_conversation = [
    # Start conversation with an audio query
    {"role": "user", "message_type": "audio", "content": "test_audios/qa_example.wav"}
]

# Generate both audio and text output
wav_output, text_output = model.generate(messages_conversation, **sampling_params, output_type="both")

# Save the generated audio
output_audio_path = "output_audio.wav"
sf.write(output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000) # Assuming 24kHz output
print(f">>> Conversational Output Audio saved to: {output_audio_path}")
print(">>> Conversational Output Text: ", text_output) # Expected output: "当然可以，这很简单。一二三四五六七八九十。"

# --- 5. Example 3: Audio-to-Audio/Text Conversation with Multiturn ---

messages = [
    {"role": "user", "message_type": "audio", "content": "test_audios/multiturn/case2/multiturn_q1.wav"},
    # This is the first turn output of Kimi-Audio
    {"role": "assistant", "message_type": "audio-text", "content": ["test_audios/multiturn/case2/multiturn_a1.wav", "当然可以，这很简单。一二三四五六七八九十。"]},
    {"role": "user", "message_type": "audio", "content": "test_audios/multiturn/case2/multiturn_q2.wav"}
]
wav, text = model.generate(messages, **sampling_params, output_type="both")


# Generate both audio and text output
wav_output, text_output = model.generate(messages_conversation, **sampling_params, output_type="both")

# Save the generated audio
output_audio_path = "output_audio.wav"
sf.write(output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000) # Assuming 24kHz output
print(f">>> Conversational Output Audio saved to: {output_audio_path}")
print(">>> Conversational Output Text: ", text_output) # Expected output: "没问题，继续数下去就是十一十二十三十四十五十六十七十八十九二十。"

print("Kimi-Audio inference examples complete.")

