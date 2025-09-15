from data.base import make_Validation_loader
import numpy as np
from tqdm import tqdm, trange
from utils import cal_EM_F1
from MOE_model.make_model import make_main_model, replace_layer
from MOE_model.hypernetwork import EnhancedHyperNetwork
import hydra
import torch
import json
from utils import get_sent_embeddings, retrieve_facts, get_word
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from train import get_word
import os


def question_decomposition(prompt, model, tok):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    outputs_new_part = outputs[0][len(inputs['input_ids'][0]):]
    if outputs_new_part[0] == tok.eos_token_id:
        return None, None
    else:
        output = tok.decode(outputs_new_part, skip_special_tokens=True)
        new_prompt = output.split('Sub-answer:')[0]
        subquestion = new_prompt.split('Sub-question:')[1].strip()
        return subquestion, prompt+"Sub-question: {}".format(subquestion)



metrics = {"EM": [], "F1": []}


def param_flatten(delta):
    delta_params = torch.cat([delta[i].flatten(start_dim=0) for i in range(len(delta))], dim=0)  # shape: [total_params]
    return delta_params


def create_pre_hook_fn(id, weights, delta):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights, delta)
    return pre_hook_fn


def model_generation_subanswer(input, input_id, model, tok):
    input_id['input_ids'] = input_id['input_ids'].cuda()
    output = model.generate(**input_id,
                            max_new_tokens=10,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                            eos_token_id=tok.eos_token_id,
                            temperature=1.0,
                            top_p=1.0,)
    predict = tok.decode(output[0], skip_special_tokens=True)
    return predict[len(input):].strip()



def gram_schmidt(A, B):
    """
    将 B 投影到 A 的正交补空间上
    A, B: [d, r]
    返回 B 在正交补上的矩阵 [d, r]
    """
    Qa, _ = torch.linalg.qr(A)   # [d, r]
    proj = Qa @ (Qa.T @ B)
    B_orth = B - proj
    return B_orth



def fuse_weights(A, B):
    """
    融合两个低秩矩阵 A, B → 输出一个融合后的 [d, r] 矩阵
    """
    B_orth = gram_schmidt(A, B)  # [d, r]
    return A + B_orth



def valid(config , hypernetwork, model, tok, valid_loader, retriever, retriever_tok, decomposer, decomposer_tok):
    hypernetwork.eval()
    model.eval()
    facts = []
    with open(config.test_dataset.format(config.data_name)) as fp:
        datas = json.load(fp)
    for data in datas:
        facts.extend(data['facts'])
    facts = list(set(facts))
    print("facts length:", len(facts))
    embs = get_sent_embeddings(facts, retriever, retriever_tok)

    for tuples in tqdm(valid_loader, desc="Valid"):
        question, answers, _ = tuples
        initial_prompt = 'Decompose the following question into sub-questions:\n{}\n'.format(question)
        split_index = len(initial_prompt)
        ix = 0
        try:
            while True:
                subquestion, prompt = question_decomposition(initial_prompt, decomposer, decomposer_tok)
                if not subquestion:
                    base_input = initial_prompt.strip() + '\nQuestion: {}\nAnswer:'.format(question)
                    base_input = base_input[split_index:]
                    base_input_token = {k: v.cuda() for k, v in tok(base_input, return_tensors="pt").items()}
                    input_embeds = model.model.embed_tokens(base_input_token['input_ids'])
                    delta = hypernetwork(input_embeds)
                    fuse_delta = []
                    for A, B in zip(use_delta, delta):
                        fuse_delta.append(fuse_weights(A, B))
                    use_delta = fuse_delta
                    inference_hook = (model.model.layers[layer_index].mlp.
                                      register_forward_pre_hook(create_pre_hook_fn(0, [1], param_flatten(use_delta))))
                    final_answer = model_generation_subanswer(base_input, base_input_token, model, tok)
                    inference_hook.remove()
                    break
                fact_ids = retrieve_facts(subquestion, embs, retriever, retriever_tok)
                fact = facts[fact_ids[0]]
                tok_fact = {k: v.cuda() for k, v in tok(fact, return_tensors="pt").items()}
                input_embeds = model.model.embed_tokens(tok_fact['input_ids'])  # shape(batchsize, length, embedding_dim:4096)
                for layer_index in config.single_layer:
                    delta = hypernetwork(input_embeds)
                    if ix == 0:
                        use_delta = delta
                        inference_hook = (model.model.layers[layer_index].mlp.
                                          register_forward_pre_hook(create_pre_hook_fn(0, [1], param_flatten(use_delta))))
                    else:
                        fuse_delta = []
                        for A,B in zip(use_delta, delta):
                            fuse_delta.append(fuse_weights(A, B))
                        use_delta = fuse_delta
                        inference_hook = (model.model.layers[layer_index].mlp.
                                          register_forward_pre_hook(create_pre_hook_fn(0, [1], param_flatten(use_delta))))
                base_input = 'Passage: {}\nQuestion: {}\nAnswer:'.format(fact, subquestion)
                base_input_token = {k: v.cuda() for k, v in tok(base_input, return_tensors="pt").items()}
                sub_answer = model_generation_subanswer(base_input, base_input_token, model, tok)
                inference_hook.remove()
                initial_prompt = prompt + '\n' + 'Sub-answer: {}\n'.format(sub_answer)
                ix+=1
        except Exception as e:
            final_answer = ''
        print(final_answer)
        EM, F1 = cal_EM_F1(final_answer, answers)
        for key, value in zip(metrics.keys(), [EM, F1]):
            metrics[key].append(value)
            print(key, len(metrics[key]), np.mean(metrics[key]) * 100)
    return metrics



@hydra.main(config_path="config", config_name="config")
def main(config):
    retriever = AutoModel.from_pretrained(config.retrieval_model_ckpt).cuda()
    retriever_tok = AutoTokenizer.from_pretrained(config.retrieval_model_ckpt)
    decomposer = AutoModelForCausalLM.from_pretrained(config.decompose_model_ckpt.format(config.data_name), device_map="auto")
    decomposer_tok = AutoTokenizer.from_pretrained(config.decompose_model_ckpt.format(config.data_name))

    hypernetwork = EnhancedHyperNetwork(embed_dim=config.embed_feature,
                                        rank=config.rank,
                                        input_dim=config.in_feature,
                                        hidden_dim=config.hid_feature,
                                        output_dim=config.out_feature).cuda()
    state_dict = torch.load(config.hypernetwork_ckpt.format(config.model_name.split("/")[-1]+'_'+config.data_name,
                                               ','.join([str(layer_index) for layer_index in config.single_layer])))
    hypernetwork.load_state_dict(state_dict)

    model, tok = make_main_model(config)
    for layer_index in config.single_layer:
        original_layer = model.model.layers[layer_index].mlp
        replace_layer(config, model, original_layer, layer_index)

    for name, param in model.named_parameters():
        param.requires_grad = False
    #
    valid_loader = make_Validation_loader(config, tok)
    metrics = valid(config , hypernetwork, model, tok, valid_loader, retriever, retriever_tok, decomposer, decomposer_tok)


if __name__ == '__main__':
    main()