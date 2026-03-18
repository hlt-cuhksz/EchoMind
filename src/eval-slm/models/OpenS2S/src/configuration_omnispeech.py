"""BLAP config"""

from transformers import PretrainedConfig
from transformers import logging
from peft import LoraConfig

from transformers.models.auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

class OmniSpeechConfig(PretrainedConfig):
    def __init__(
        self, 
        audio_encoder_config=None,
        llm_config=None,
        tts_lm_config=None,
        lora_config=None,
        conv_kernel_sizes="5,5",
        adapter_inner_dim=512,
        interleave_strategy="1:2",
        **kwargs
    ):
        if isinstance(audio_encoder_config, dict):
            if "model_type" in audio_encoder_config:
                pass
            else:
                logger.info("audio encoder config is None. Initializing with qwen2_audio_encoder")
                audio_encoder_config["model_type"] = "qwen2_audio_encoder"
        elif audio_encoder_config is None:
            logger.info("audio encoder config is None. Initializing with qwen2_audio_encoder")
            audio_encoder_config = {
                "model_type": "qwen2_audio_encoder",
                "d_model": 1280,
                "encoder_attention_heads": 20,
                "encoder_ffn_dim": 5120,
                "encoder_layerdrop": 0.0,
                "encoder_layers": 32,
                "num_mel_bins": 128,
                "max_source_positions": 1500,
                "scale_embedding": False,
                "activation_function": "gelu",
            }
        else:
            raise NotImplementedError


        if isinstance(llm_config, dict):
            if "model_type" in llm_config:
                pass
            else:
                logger.info("llm config is None. Initializing with qwen3")
                llm_config["model_type"] = "qwen3"
        elif llm_config is None:
            logger.info("llm config is None. Initializing with qwen3")
            llm_config = {
                "model_type": "qwen3"
            }
        else:
            raise NotImplementedError
        
        if isinstance(tts_lm_config, dict):
            if "model_type" in tts_lm_config:
                pass
            else:
                logger.info("tts lm config is None. Initializing with qwen3")
                tts_lm_config["model_type"] = "qwen3"
        elif tts_lm_config is None:
            logger.info("tts lm config is None. Initializing with qwen3")
            tts_lm_config = {
                "model_type": "qwen3"
            }

        self.audio_encoder_config = CONFIG_MAPPING[audio_encoder_config["model_type"]](**audio_encoder_config)
        self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        self.tts_lm_config = CONFIG_MAPPING[tts_lm_config["model_type"]](**tts_lm_config)

        self.lora_config = lora_config

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim
        self.interleave_strategy = interleave_strategy

        super().__init__(**kwargs)