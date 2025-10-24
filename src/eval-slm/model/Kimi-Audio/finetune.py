# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca and QwenLM/Qwen.


from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, Optional

import torch
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoTokenizer
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from accelerate.utils import DistributedType
from huggingface_hub import snapshot_download

from finetune_codes.model import KimiAudioModel
from finetune_codes.datasets import LazySupervisedDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="moonshotai/Kimi-Audio-7B")
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )



def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)




def make_supervised_data_module(
    whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading data...")

    with open(data_args.data_path, "r") as f:
        lines = f.readlines()
        all_data = [json.loads(line) for line in lines]

    if data_args.eval_ratio > 0:
        eval_data = all_data[:int(len(all_data) * data_args.eval_ratio)]
        train_data = all_data[int(len(all_data) * data_args.eval_ratio):]
        assert len(eval_data) > 0, "No evaluation data found"
        assert len(train_data) > 0, "No training data found"
    else:
        eval_data = None
        train_data = all_data

    train_dataset = dataset_cls(train_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)

    if eval_data:
        eval_dataset = dataset_cls(eval_data, whisper_model=whisper_model, text_tokenizer=text_tokenizer, max_len=max_len, kimia_token_offset=kimia_token_offset)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def compute_loss(outputs, labels, num_items_in_batch=None):

    audio_logits, text_logits = outputs.logits

    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    assert audio_labels.shape[0] == 1, print("we only support micro batch size 1 for demo purpose")

    audio_loss = torch.nn.functional.cross_entropy(audio_logits.view(-1, audio_logits.shape[-1]), audio_labels.view(-1), reduction="none")
    text_loss = torch.nn.functional.cross_entropy(text_logits.view(-1, text_logits.shape[-1]), text_labels.view(-1), reduction="none")


    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (audio_loss_mask.view(-1).sum() + 1e-4)
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (text_loss_mask.view(-1).sum() + 1e-4)
    loss = audio_loss + text_loss
    return loss


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    logger.info(f"Loading kimi-audio main model")

    if os.path.exists(model_args.model_name_or_path):
        # local path
        cache_path = model_args.model_name_or_path
    else:
        # cache everything if model_path is a model-id
        cache_path = snapshot_download(model_args.model_name_or_path)

    logger.info(f"Looking for resources in {cache_path}")
    # check if model_path exists
    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")
    model = KimiAudioModel.from_pretrained(model_args.model_path, 
                                           device_map=None,
                                           **model_load_kwargs)

    text_tokenizer = AutoTokenizer.from_pretrained(
        cache_path, trust_remote_code=True
    )

    # Load data
    data_module = make_supervised_data_module(
        whisper_model=model.whisper_model, text_tokenizer=text_tokenizer,
        data_args=data_args, max_len=training_args.model_max_length, kimia_token_offset=model.config.kimia_token_offset
    )

    # Start trainner
    trainer = Trainer(
        model=model, args=training_args, 
        compute_loss_func=compute_loss,
        data_collator=data_module["train_dataset"].collate_fn,
        **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()