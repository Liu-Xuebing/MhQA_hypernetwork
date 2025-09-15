from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .ExperModel import MoE, ParallelFFNMoE

def make_main_model(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='auto', trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    if config.half:
        model.bfloat16()
    for name, param in model.named_parameters():
        param.requires_grad = False
    return model, tokenizer


def replace_layer(config, model, original_layer, layer_index):
    moes = MoE(config.in_feature, config.hid_feature, config.out_feature, config.rank, num_experts=1)
    if "Llama" in config.model_name:
        model.model.layers[layer_index].mlp = ParallelFFNMoE(original_layer, moes).to(next(original_layer.parameters()).device)
    elif "gpt" in config.model_name:
        model.transformer.h[layer_index].mlp = ParallelFFNMoE(original_layer, moes).to(next(original_layer.parameters()).device)