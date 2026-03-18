from kimia_infer.api.kimia import KimiAudio


model = KimiAudio(model_path="output/finetuned_hf_for_inference", load_detokenizer=False)


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

messages = [
    {"role": "user", "message_type": "text", "content": "Please transcribe the spoken content into written text."},
    {
        "role": "user",
        "message_type": "audio",
        "content": "finetune_codes/demo_data/audio_understanding/audios/librispeech_1263-139804-0001.flac",
    },
]

wav, text = model.generate(messages, **sampling_params, output_type="text")
print(">>> output text: ", text)








