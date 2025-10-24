import argparse
import itertools
import json
import os
import random
import re
import sys
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path

import torch
import tqdm
from datasets import load_dataset
from tn.english.normalizer import Normalizer as EnNormalizer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import torchaudio
from vita_audio.tokenizer import get_audio_tokenizer


def collate_fn(batches):
    input_ids = [sample["input_ids"] for sample in batches]

    refs = [sample["ref"] for sample in batches]
    filenames = [sample["filename"] for sample in batches]

    return input_ids, refs, filenames


class TTSDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, tokenizer, audio_tokenizer, default_system_message=None, add_generation_prompt=True):
        data = load_dataset("json", data_files=json_path, keep_in_memory=False)
        self.data = data["train"]

        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.default_system_message = default_system_message
        self.add_generation_prompt = add_generation_prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        messages = []

        if self.default_system_message is not None:
            messages = self.default_system_message + messages

        role = "user"
        content = sample["messages"][0]["content"]
        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
            return_tensors="pt",
        )

        ref = sample["messages"][0]["content"]
        ref = ref.replace("Convert the text to speech.\n", "")
        ref = ref.strip()

        filepath = sample["audios"][0]
        filename = os.path.basename(filepath)

        return {
            "input_ids": input_ids,
            "ref": ref,
            "filename": filename,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def inference(model, tokenizer, audio_tokenizer, dataloader, output_dir, asr_model):

    audio_offset = tokenizer.convert_tokens_to_ids("<|audio_0|>")
    en_tn_model = EnNormalizer(overwrite_cache=True)

    outputs = []

    for _, (
        batched_input_ids,
        batched_ref,
        batched_filename,
    ) in enumerate(tqdm.tqdm(dataloader)):
        for input_ids, ref, filename in zip(
            batched_input_ids, batched_ref, batched_filename
        ):

            responses = model.generate(
                input_ids=input_ids.cuda(),
                # temperature=0.2,
                # top_p=0.8,
                # do_sample=False,
                # temperature=1.0,
                max_new_tokens=1024,
                min_new_tokens=1,
            )

            response = responses[0][len(input_ids[0]) :]

            text_tokens = []
            audio_tokens = []
            for token_id in response:
                if token_id >= audio_offset:
                    audio_tokens.append(token_id - audio_offset)
                else:
                    text_tokens.append(token_id)

            if len(audio_tokens) == 0:
                continue

            tts_speech = audio_tokenizer.decode(audio_tokens)

            wav_dir = os.path.join(output_dir, "audio")
            wav_path = os.path.join(wav_dir, filename + ".wav")
            os.makedirs(os.path.dirname(wav_path), exist_ok=True)
            torchaudio.save(wav_path, tts_speech.unsqueeze(0), 22050, format="wav")

            hyp = asr_model(wav_path, return_timestamps=True)["text"].strip()

            hyp = en_tn_model.normalize(hyp)
            ref = en_tn_model.normalize(ref)

            hyp = re.sub(r"\W+", " ", hyp)
            ref = re.sub(r"\W+", " ", ref)

            outputs.append((hyp, ref))

            print("")
            print("=" * 100)
            # print(f"{len(input_id)=}")
            # print(f"{len(response)=}")
            print(f"{tokenizer.decode(response, skip_special_tokens=False)}")
            print(f"{filename=}")

    return outputs


def load_asr_model():
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    rank = torch.distributed.get_rank()
    device = f"cuda:{rank}"
    torch_dtype = torch.float16

    model_id = "/data/models/openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model_name_or_path", type=str, required=True, help="model_name_or_path")
    parser.add_argument(
        "--audio_tokenizer_path", type=str, required=True, help="audio_tokenizer_path"
    )
    parser.add_argument(
        "--audio_tokenizer_type", type=str, required=True, help="audio_tokenizer_type"
    )
    parser.add_argument("--flow_path", type=str, required=True, help="flow_path")

    parser.add_argument("--json_path", type=str, required=True, help="json_path")
    parser.add_argument("--output_dir", type=str, required=True, help="output_dir")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--speaker_prompt", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    print(f"{args=}")

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
        timeout=timedelta(seconds=7200),
    )

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    random.seed(42)
    torch.manual_seed(42)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    # ================================================================
    if "glm" in config.model_type.lower():
        from get_chat_template import glm4_chat_template as chat_template

        add_generation_prompt = True

        default_system_message = [
            {
                "role": "system",
                "content": "User will provide you with a speech instruction. Do it step by step. First, think about the instruction and respond in a interleaved manner, with 13 text token followed by 26 audio tokens.",
            }
        ]

    if "qwen2" in config.model_type.lower():
        from get_chat_template import qwen2_chat_template as chat_template

        add_generation_prompt = True

        default_system_message = []

    if "hunyuan" in config.model_type.lower():
        from get_chat_template import hunyuan_chat_template as chat_template

        add_generation_prompt = False

        default_system_message = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            }
        ]

    # ================================================================
    print("Loading model")
    device = "cuda"
    # device_map = "auto"
    device_map = "cuda"
    # torch_dtype=torch.float16
    torch_dtype = torch.bfloat16

    rank = torch.distributed.get_rank()

    audio_tokenizer = get_audio_tokenizer(
        args.audio_tokenizer_path, args.audio_tokenizer_type, flow_path=args.flow_path, rank=rank
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        chat_template=chat_template,
    )
    # print("tokenizer", tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    ).eval()
    # print("model", model)

    model.generation_config = GenerationConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )

    model.generation_config.max_new_tokens = 4096
    model.generation_config.chat_format = "chatml"
    model.generation_config.max_window_size = 8192
    model.generation_config.use_cache = True
    model.generation_config.do_sample = True
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    if model.config.model_type == "hunyuan":
        model.generation_config.eos_token_id = tokenizer.eos_id

    asr_model = load_asr_model()

    # ================================================================
    print("Loading data")
    dataset = TTSDataset(
        json_path=args.json_path,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        default_system_message=default_system_message,
        add_generation_prompt=add_generation_prompt,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(
            collate_fn,
        ),
    )

    # ================================================================
    outputs = inference(model, tokenizer, audio_tokenizer, dataloader, args.output_dir, asr_model)

    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:
        # json_name = Path("_".join(os.path.normpath(args.json_path).split(os.sep)[-2:])).stem
        json_name = Path(os.path.normpath(args.json_path).split(os.sep)[-1]).stem
        hyp_path = os.path.join(args.output_dir, f"{json_name}_hyp.txt")
        ref_path = os.path.join(args.output_dir, f"{json_name}_ref.txt")

        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        os.makedirs(os.path.dirname(hyp_path), exist_ok=True)

        hyp_file = open(hyp_path, "w")
        ref_file = open(ref_path, "w")

        for sample_idx, (hyp, ref) in enumerate(merged_outputs):
            hyp_file.write(f"{sample_idx} {hyp}" + "\n")
            ref_file.write(f"{sample_idx} {ref}" + "\n")

        hyp_file.close()
        ref_file.close()

        hyp_ref_path = os.path.join(args.output_dir, f"{json_name}_hyp_ref.json")
        hyp_ref_file = open(hyp_ref_path, "w")
        json.dump(merged_outputs, hyp_ref_file, indent=4)
        hyp_ref_file.close()

    torch.distributed.barrier()
    print("Done.")
