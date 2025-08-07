import os,json
from turtledemo.penrose import start

from data.base import make_Validation_loader
import numpy as np
from tqdm import tqdm, trange
from utils import cross_entropy, cal_EM_F1, tansfer_to_scftmax
from MOE_model.make_model import make_model, replace_layer
from MOE_model.hypernetwork import HyperNetwork
import hydra
import torch
import time



metrics = {
        "base_EM": [],
        "base_F1": [],
        "dir_EM": [],
        "dir_F1": [],}
        # "pro_EM": [],
        # "pro_F1": []}

editing_time = []
inference_time = []
all_time = []


def create_pre_hook_fn(id, weights, delta):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights, delta)
    return pre_hook_fn


def EM_F1_of_model_generation(input_id, input_id_len, model, tok, answers):
    output = model.generate(input_id, max_new_tokens=10, temperature=0.0, top_p=1.0, top_k=-1,
                                 do_sample=False)
    predict = tok.decode(output[0], skip_special_tokens=True)
    EM, F1 = cal_EM_F1(predict[len(input_id_len):].strip().split('\n')[0], answers)
    return EM, F1



def valid_o(original_model, tok, valid_loader):
    original_model.eval()
    for tuples in tqdm(valid_loader, desc="Valid"):
        base_input, len_base_input, direct_input, len_direct_input, prompt_input, len_prompt_input, sentences, answers, activate_sentence = tuples
        with torch.no_grad():
            base_EM, base_F1 = EM_F1_of_model_generation(base_input, len_base_input, original_model, tok, answers)
            dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input, len_direct_input, original_model, tok, answers)
            # pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input, len_prompt_input, original_model, tok, answers)

        for key, value in zip(metrics.keys(), [base_EM, base_F1, dir_EM, dir_F1]):
            metrics[key].append(value)
        for key, value in metrics.items():
            print(key, len(metrics[key]), np.mean(metrics[key]) * 100)

    return metrics



def valid(config, hypernetwork, model, tok, valid_loader, weights):
    hypernetwork.eval()
    model.eval()

    for tuples in tqdm(valid_loader, desc="Valid"):
        weights = tansfer_to_scftmax(weights)
        base_input, len_base_input, direct_input, len_direct_input, prompt_input, len_prompt_input, sentences, answers, activate_sentences = tuples

        deltas = []
        for activate_sentence in activate_sentences:
            start_time = time.time()
            input_embeds = model.model.embed_tokens(activate_sentence['input_ids'])  # shape(batchsize, length, embedding_dim:4096)
            delta = hypernetwork(input_embeds)
            deltas.append(delta)
            end_time = time.time()
            elapsed_time = end_time - start_time
            editing_time.append(elapsed_time)


        with torch.no_grad():
            hook_base = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(
                create_pre_hook_fn(0, weights, deltas))
            base_EM, base_F1 = EM_F1_of_model_generation(base_input, len_base_input, model, tok, answers)
            hook_base.remove()


            hook_RAG = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(
                create_pre_hook_fn(direct_input.size(1) - base_input.size(1), weights, deltas))
            dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input, len_direct_input, model, tok, answers)
            hook_RAG.remove()


            # hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(prompt_input.size(1)-base_input.size(1), weights, delta))
            # pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input, len_prompt_input, model, tok, answers)
            # hook.remove()

        for key, value in zip(metrics.keys(), [base_EM, base_F1, dir_EM, dir_F1]):
            metrics[key].append(value)
        for key, value in metrics.items():
            print(key, len(metrics[key]), np.mean(metrics[key]) * 100)

    return metrics



@hydra.main(config_path="config", config_name="config")
def main(config):
    hypernetwork = HyperNetwork(config.embed_feature, config.rank, config.hid_feature, config.out_feature, config.in_feature).cuda()
    hypernetwork.load_state_dict(torch.load(config.hypernetwork_ckpt.format(config.model_name.split("/")[-1], config.data_name, config.single_layer)))  # 加载参数

    model, tok = make_model(config)
    # original_layer = model.model.layers[config.single_layer].mlp
    # replace_layer(config, model, original_layer, config.num_experts)

    for name, param in model.named_parameters():
        param.requires_grad = False

    valid_loader = make_Validation_loader(config, tok)
    # metrics = valid(config , hypernetwork, model, tok, valid_loader,[1.0]*config.num_experts)
    metrics = valid_o(model, tok, valid_loader)


    for i in range(6):
        start = i * 250
        end = start + 250
        print(sum(metrics['base_EM'][start:end]) / 250)
        print(sum(metrics['base_F1'][start:end]) / 250)
        print(sum(metrics['dir_EM'][start:end]) / 250)
        print(sum(metrics['dir_F1'][start:end]) / 250)
        print('\n')




if __name__ == '__main__':
    main()