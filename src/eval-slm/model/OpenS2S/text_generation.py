import logging
import os
import sys
import json
import fire
import random
from typing import Any, Dict, List, Optional, Union, Tuple
import math
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("text generation")


instruction_dict = {
    "asr": {
        "generation": "Continue the following sentence in a coherent style: {text}",
        "training": "Continue the following sentence in a coherent style: ",
    },
    "ser": {
        "generation": "Continue the following sentence that reflects a '{label}' emotion tone in a coherent style: {text}",
        "training": "Continue the following sentence based on the conveyed emotion tone in a coherent style: "
    },
}
TASK_WO_LABEL = [
    "asr",
]
TASK_WITH_LABEL = [
    "ser",
]


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} "
        f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_dataset(manifest, nshard, rank):
    with open(manifest, "r") as f:
        lines = f.readlines()
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]
        lines = [json.loads(line.strip()) for line in lines]

    return lines


def apply_tokenizer(lines, tokenizer, task):
    dataset = []
    for line in lines:
        text = line["text"]
        audio = line["audio"]

        if task in TASK_WO_LABEL:
            generation_instruction = instruction_dict[task]["generation"].format(text=text)
            training_instruction = instruction_dict[task]["training"]
        elif task in TASK_WITH_LABEL:
            assert line["label"]
            generation_instruction = instruction_dict[task]["generation"].format(label=line["label"], text=text)
            training_instruction = instruction_dict[task]["training"].format(label=line["label"])
        else:
            raise NotImplementedError

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": generation_instruction}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        content = [
            {"text": training_instruction, "audio": "", "speech_units": "", "spk_emb": ""},
            {"text": "", "audio": audio, "speech_units": "", "spk_emb": ""}
        ]

        line["prompt"] = prompt
        line["content"] = content
        dataset.append(line)

    return dataset


def generate(
    llm_path,
    manifest,
    lab_dir,
    task="asr",
    nshard=4,
    rank=0,
    batch_size=1000,
    max_tokens=1024,
):
    lines = get_dataset(manifest, nshard, rank)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    dataset = apply_tokenizer(lines, tokenizer, task)

    split = os.path.splitext(os.path.basename(manifest))[0]
    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.jsonl"
    os.makedirs(lab_dir, exist_ok=True)

    sampling_params = SamplingParams(
        top_k=1, ### greedy
        max_tokens=max_tokens,
    )
    llm = LLM(model=llm_path)
    
    num_batch = math.ceil(1.0 * len(dataset) / batch_size)
    progress_bar = tqdm(range(num_batch), disable=False)
    with open(lab_path, "w") as f:
        for idx in range(0, len(dataset), batch_size):
            batch = dataset[idx: idx+batch_size]
            text = [b["prompt"] for b in batch]

            outputs = llm.generate(text, sampling_params)
            for example, output in zip(batch, outputs):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                content = example["content"]
                messages = [
                    {
                        "role": "user", "content": content
                    },
                    {
                        "role": "assistant", "content": [{"text": generated_text, "audio": "", "speech_units": "", "spk_emb": ""}]
                    }
                ]
                json_string = json.dumps(
                    {
                        "messages": messages,
                    },
                    ensure_ascii=False
                )
                print(json_string, file=f, flush=True)
            progress_bar.update(1)

    logger.info("finished successfully")
    
if __name__ == "__main__":
    fire.Fire({
        'generate': generate,
    })