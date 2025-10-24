audio_path = "介绍一下上海.wav"
# audio_path = "piano.mp3"
# audio_path = "hey-ya.opus"
# audio_path = "snoop.opus"
# audio_path = "swan.opus"
# audio_path = "青花瓷人声.mp3"
# audio_path = "恭喜你发财人声.mp3"


def encodec():
    from datasets import load_dataset, Audio
    from transformers import EncodecModel, AutoProcessor

    # load a demonstration datasets
    librispeech_dummy = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )

    # load the model + processor (for pre-processing the audio)
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    # cast the audio data to the correct sampling rate for the model
    librispeech_dummy = librispeech_dummy.cast_column(
        "audio", Audio(sampling_rate=processor.sampling_rate)
    )
    audio_sample = librispeech_dummy[0]["audio"]["array"]
    sampling_rate = librispeech_dummy[0]["audio"]["sampling_rate"]
    print(f"len(audio_sample) / sampling_rate {len(audio_sample) / sampling_rate}")

    # pre-process the inputs
    inputs = processor(
        raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt"
    )

    # explicitly encode then decode the audio inputs
    encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=6)
    print(f"encoder_outputs {encoder_outputs} {encoder_outputs.audio_codes.size()}")
    audio_values = model.decode(
        encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"]
    )[0]
    print(f"audio_values {audio_values} {audio_values.size()}")

    # or the equivalent with a forward pass
    audio_values = model(inputs["input_values"], inputs["padding_mask"], bandwidth=6).audio_values
    print(f"audio_values {audio_values} {audio_values.size()}")

    return

    from encodec import EncodecModel
    from encodec.utils import convert_audio

    import torchaudio
    import torch

    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
    model.set_target_bandwidth(6.0)

    # Load and pre-process the audio waveform
    wav, sr = torchaudio.load(audio_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)

    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = model.encode(wav)
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    print(f"codes {codes.size()}")


def xcodec2():
    import torch
    import soundfile as sf
    from transformers import AutoConfig

    from xcodec2.modeling_xcodec2 import XCodec2Model

    model_path = "/data/models/HKUSTAudio/xcodec2"

    model = XCodec2Model.from_pretrained(model_path)
    model.eval().cuda()

    wav, sr = sf.read(audio_path)
    print(f"len(wav) / sr {len(wav) / sr}")
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)  # Shape: (1, T)

    with torch.no_grad():
        # Only 16khz speech
        # Only supports single input. For batch inference, please refer to the link below.
        vq_code = model.encode_code(input_waveform=wav_tensor)
        print("Code:", vq_code, vq_code.size())

        recon_wav = model.decode_code(vq_code).cpu()  # Shape: (1, 1, T')

        sf.write(audio_path[:-4] + "_reconstructed_xcodec2.wav", recon_wav[0, 0, :].numpy(), sr)
    print("Done! Check reconstructed.wav")


def glm4voice():
    import sys
    import torchaudio

    sys.path.insert(0, "third_party/GLM-4-Voice/")
    sys.path.insert(0, "third_party/GLM-4-Voice/cosyvoice/")
    sys.path.insert(0, "third_party/GLM-4-Voice/third_party/Matcha-TTS/")

    audio_tokenizer_path = "/data/models/THUDM/glm-4-voice-tokenizer/"
    audio_tokenizer_type = "glm4voice"
    flow_path = "/data/models/THUDM/glm-4-voice-decoder/"
    rank = 0
    from vita_audio.tokenizer import get_audio_tokenizer

    audio_tokenizer = get_audio_tokenizer(
        audio_tokenizer_path, audio_tokenizer_type, flow_path=flow_path, rank=rank
    )

    audio_tokens = audio_tokenizer.encode(audio_path)
    print(f"audio_tokens {audio_tokens} {len(audio_tokens)}")
    tts_speech = audio_tokenizer.decode(audio_tokens)
    torchaudio.save(
        audio_path[:-4] + "_reconstructed_glm4voice.wav",
        tts_speech.unsqueeze(0),
        22050,
        format="wav",
    )


def cosyvoice2():

    audio_path = "介绍一下上海.wav"

    import torch
    import torchaudio
    import sys

    sys.path.insert(0, "third_party/CosyVoice/")
    sys.path.insert(0, "third_party/CosyVoice/third_party/Matcha-TTS/")

    audio_tokenizer_path = "/data/models/FunAudioLLM/CosyVoice2-0.5B/"
    audio_tokenizer_type = "cosyvoice2"
    rank = 0
    from vita_audio.tokenizer import get_audio_tokenizer

    audio_tokenizer = get_audio_tokenizer(audio_tokenizer_path, audio_tokenizer_type, rank=rank)

    audio_tokens = audio_tokenizer.encode(audio_path)

    import tqdm
    import time

    start = time.time()

    for _ in tqdm.tqdm(range(100)):
        # speech, sample_rate = torchaudio.load(audio_path)
        # speech = speech[:, sample_rate * 30 : sample_rate * 60]
        # torchaudio.save("tmp.wav", speech, sample_rate, format="wav")
        # audio_path = "tmp.wav"

        audio_tokens = audio_tokenizer.encode(audio_path)
        print(f"audio_tokens {audio_tokens} {len(audio_tokens)}")
        tts_speech = audio_tokenizer.decode(audio_tokens)
        torchaudio.save(
            audio_path[:-4] + "_reconstructed_cosyvoice2.wav",
            tts_speech.unsqueeze(0),
            audio_tokenizer.cosyvoice.sample_rate,
            format="wav",
        )

    end = time.time()
    print(end - start)


def dac():
    import dac
    from audiotools import AudioSignal

    # Download a model
    model_path = dac.utils.download(model_type="44khz")
    model = dac.DAC.load(model_path)

    model.to("cuda")

    # Load audio signal file
    signal = AudioSignal(audio_path)

    # Encode audio signal as one long file
    # (may run out of GPU memory on long files)
    signal.to(model.device)

    x = model.preprocess(signal.audio_data, signal.sample_rate)
    z, codes, latents, _, _ = model.encode(x)

    # Decode audio signal
    y = model.decode(z)

    # Alternatively, use the `compress` and `decompress` functions
    # to compress long files.

    signal = signal.cpu()
    x = model.compress(signal)
    print(f"x {x} {x.codes.size()}")

    # Save and load to and from disk
    x.save("compressed.dac")
    x = dac.DACFile.load("compressed.dac")

    # Decompress it back to an AudioSignal
    y = model.decompress(x)

    # Write to file
    y.write(audio_path[:-4] + "_reconstructed_dac.wav")


def snac():
    from snac import SNAC
    import torchaudio

    if torch.cuda.is_available():
        model = SNAC.from_pretrained("/data/models/hubertsiuzdak/snac_24khz").eval().cuda()
    else:
        model = SNAC.from_pretrained("/data/models/hubertsiuzdak/snac_24khz").eval()
    # audio = torch.randn(1, 1, 32000).cuda()  # placeholder for actual audio with shape (B, 1, T)
    audio, sampling_rate = torchaudio.load(audio_path)
    print(f"audio {audio.size()}")
    print(f"sampling_rate {sampling_rate}")
    print(f"len(audio) / sampling_rate {len(audio[0]) / sampling_rate}")

    print(f"model.sampling_rate {model.sampling_rate}")
    audio = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=model.sampling_rate)(
        audio
    )
    audio = audio.unsqueeze(0)
    print(f"audio {audio.size()}")
    if torch.cuda.is_available():
        audio = audio.to("cuda")

    with torch.inference_mode():
        codes = model.encode(audio)
        print(f"codes {codes} {[x.size() for x in codes]}")

        codes = shift_code(codes, model.codebook_size, model.vq_strides)
        print(f"codes {codes} {codes.size()}")

        codes = inverse_shift_code(codes, model.codebook_size, model.vq_strides)
        print(f"codes {codes} {[x.size() for x in codes]}")

        audio_hat = model.decode(codes)
    print(f"audio_hat {audio_hat.size()}")

    torchaudio.save(
        audio_path.split(".")[0] + "_reconstructed_snac.wav",
        audio_hat.squeeze(0).cpu(),
        model.sampling_rate,
        format="wav",
    )


def shift_code(codes, codebook_size, vq_strides):
    # codes: [torch.Size([1, 43]), torch.Size([1, 86]), torch.Size([1, 172])]

    # codes = torch.cat([x.reshape(1, -1, vq_strides[-i-1]) + i * codebook_size for i, x in enumerate(codes)], dim=-1).reshape(-1)

    codes = [x.reshape(1, -1, s) for s, x in zip(vq_strides[::-1], codes)]
    codes = torch.cat(
        [
            x + i * codebook_size
            for i, x in enumerate(torch.cat(codes, dim=-1).chunk(sum(vq_strides), dim=-1))
        ],
        dim=-1,
    ).reshape(-1)

    return codes


def inverse_shift_code(codes, codebook_size, vq_strides):
    # codes: torch.Size([301])

    # codes = [x.reshape(1, -1) - i * codebook_size for i, x in enumerate(codes.reshape(1, -1, sum(vq_strides)).split(vq_strides[::-1], dim=-1))]

    codes = torch.cat(
        [
            x - i * codebook_size
            for i, x in enumerate(
                codes.reshape(1, -1, sum(vq_strides)).chunk(sum(vq_strides), dim=-1)
            )
        ],
        dim=-1,
    ).split(vq_strides[::-1], dim=-1)
    codes = [x.reshape(1, -1) for x in codes]

    return codes


def sparktts():
    import torch
    import torchaudio
    import soundfile
    import sys

    sys.path.insert(0, "/vita_audio/third_party/Spark-TTS/")
    from sparktts.models.audio_tokenizer import BiCodecTokenizer

    model_dir = "/data/models/SparkAudio/Spark-TTS-0.5B"
    device = "cuda:0"
    audio_tokenizer = BiCodecTokenizer(model_dir, device=device)

    global_token_ids, semantic_token_ids = audio_tokenizer.tokenize(audio_path)

    global_token_ids, _ = audio_tokenizer.tokenize(
        "/data/vita_audio/third_party/CosyVoice/asset/zero_shot_prompt.wav"
    )
    global_token_ids = torch.zeros((1, 1, 32), dtype=torch.long)

    print(f"{global_token_ids.size()}")
    print(f"{semantic_token_ids.size()}")

    wav = audio_tokenizer.detokenize(
        global_token_ids.to(device).squeeze(0),
        semantic_token_ids.to(device),
    )

    wav = torch.tensor(wav).unsqueeze(0)

    print(f"{wav.shape}")

    torchaudio.save(
        audio_path.split(".")[0] + "_reconstructed_sparktts.wav",
        wav,
        audio_tokenizer.config["sample_rate"],
        format="wav",
    )
    # torchaudio.save(audio_path.split(".")[0] + "_reconstructed_sparktts.wav", wav, 16000, format="wav")
    # soundfile.write(audio_path.split(".")[0] + "_reconstructed_sparktts.wav", wav, samplerate=16000)


def stable_codec():
    import torch
    import torchaudio
    from stable_codec import StableCodec

    model = StableCodec(
        model_config_path="/data/models/stabilityai/stable-codec-speech-16k/model_config.json",
        ckpt_path="/data/models/stabilityai/stable-codec-speech-16k/model.ckpt",
        # ckpt_path="<path-to-checkpoint>", # optional, can be `None`,
        device=torch.device("cuda"),
    )

    latents, tokens = model.encode(audio_path)
    print(f"{latents.size()=}")
    print(f"{tokens.size()=}")
    print(f"{tokens=}")
    decoded_audio = model.decode(tokens)
    print(f"{decoded_audio.size()=}")

    # while True:

    torchaudio.save(
        audio_path.split(".")[0] + "_reconstructed_stable_codec.wav",
        decoded_audio.squeeze(0).cpu(),
        model.sample_rate,
    )


if __name__ == "__main__":

    # encodec()
    # xcodec2()
    # cosyvoice2()
    # glm4voice()
    # dac()
    # snac()
    # sparktts()
    stable_codec()
