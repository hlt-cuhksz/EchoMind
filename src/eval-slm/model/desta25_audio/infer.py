from desta import DeSTA25AudioModel

# Load the model from Hugging Face
model = DeSTA25AudioModel.from_pretrained("/share/workspace/EQ-SLM/EQ-Bench/models/DeSTA2.5-Audio-Llama-3.1-8B")
print("Model loaded successfully.")
model.to("cuda")
print("Model moved to GPU.")

# Run inference with audio input
messages = [
    {
        "role": "system",
        "content": "Focus on the audio clips and instructions."
    },
    {
        "role": "user",
        "content": "<|AUDIO|>\nDescribe this audio.",
        "audios": [{
            "audio": "/share/home/lvyou/models/demo.wav",
            "text": None
        }]
    }
]

outputs = model.generate(
    messages=messages,
    do_sample=False,
    top_p=1.0,
    temperature=1.0,
    max_new_tokens=512
)

print(outputs.text)
print(outputs.audios)
