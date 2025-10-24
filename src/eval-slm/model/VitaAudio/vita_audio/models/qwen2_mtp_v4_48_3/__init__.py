from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM

from .modeling_qwen2 import Qwen2MTPForCausalLM
from .configuration_qwen2 import Qwen2MTPConfig

AutoConfig.register("qwen2_mtp", Qwen2MTPConfig)
AutoModelForCausalLM.register(Qwen2MTPConfig, Qwen2MTPForCausalLM)
# AutoTokenizer.register(Qwen2MTPConfig, Qwen2MTPTokenizer)

Qwen2MTPConfig.register_for_auto_class()
# Qwen2MTPModel.register_for_auto_class("AutoModel")
Qwen2MTPForCausalLM.register_for_auto_class("AutoModelForCausalLM")
