from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .ExperModel import MoE, ParallelFFNMoE
from .diffusion_expert import DiffusionExpert

def make_model(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='auto', trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    if config.half:
        model.bfloat16()
    for name, param in model.named_parameters():
        param.requires_grad = False

    return model, tokenizer


def replace_layer(config, model, original_layer):
    moes = DiffusionExpert(config.embed_feature, config.step)
    if "Llama" in config.model_name:
        model.model.layers[config.single_layer].mlp = ParallelFFNMoE(original_layer, moes).to(next(original_layer.parameters()).device)
    elif "gpt" in config.model_name:
        model.transformer.h[config.single_layer].mlp = ParallelFFNMoE(original_layer, moes).to(next(original_layer.parameters()).device)

