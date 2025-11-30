from sympy.abc import delta

from data.base import make_Validation_loader
import numpy as np
from tqdm import tqdm, trange
from utils import cal_EM_F1, mean_pooling
from MOE_model.make_model import make_main_model, replace_layer
from MOE_model.hypernetwork import HyperKVGeneratorFixed
import hydra
import torch
import json
from utils import get_sent_embeddings, retrieve_facts, get_word
from utils import get_sent_embeddings, retrieve_facts, BM25Retriever
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import time
import os
from KV_train import cross_attention, make_simple_cross_attn_hook


metrics = {"EM": [],
           "F1": []}

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
    return predict[len(input):].split('\n')[0].strip()




def gram_schmidt_batch(A, B):
    """
    将 B 投影到 A 的正交补空间上
    A, B: [B, r, D]  # batch, rank, feature_dim
    返回 B 在正交补上的矩阵 [B, r, D]
    """
    B_orth_list = []
    for i in range(A.size(0)):  # 遍历 batch
        Ai = A[i].T  # [D, r]，qr 在列方向正交
        Bi = B[i].T  # [D, r]
        Qa, _ = torch.linalg.qr(Ai)  # [D, r]
        proj = Qa @ (Qa.T @ Bi)      # [D, r]
        Bi_orth = Bi - proj           # [D, r]
        B_orth_list.append(Bi_orth.T)  # 转回 [r, D]
    B_orth = torch.stack(B_orth_list, dim=0)  # [B, r, D]
    return B_orth


def fuse_weights_batch(A, B):
    """
    融合两个 batch 的低秩矩阵 A, B → 输出一个融合后的 [B, r, D]
    """
    B_orth = gram_schmidt_batch(A, B)  # [B, r, D]
    return A + B_orth


def mean_fuse(A, B):
    return (A + B) / 2.0

def add_fuse(A, B):
    return A + B

def concat_fuse(A, B):
    return torch.cat((A, B), dim=1)

def ties_fuse(A, B, trim_ratio=0.2):
    # Step 1: Stack A and B
    tasks = torch.stack([A, B], dim=0)  # shape: (num_tasks, ...)
    # Step 2: Keep top trim_ratio magnitude values
    abs_tasks = tasks.abs()
    flat_abs = abs_tasks.flatten()
    k = max(int((1 - trim_ratio) * flat_abs.numel()), 1)  # top trim_ratio
    if k >= flat_abs.numel():
        threshold = flat_abs.min()  # 保留全部
    else:
        threshold = torch.kthvalue(flat_abs, k).values.item()
    mask = abs_tasks >= threshold  # only keep top trim_ratio
    # Step 3: Determine sign (γm)
    signs = torch.sign((tasks.sign() * mask).sum(dim=0))
    signs[signs == 0] = 1  # tie -> +1
    # Step 4: Keep only aligned values
    aligned = tasks * mask
    aligned = torch.where(aligned.sign() == signs, aligned, torch.zeros_like(aligned))
    # Step 5: Mean of aligned values
    fused = aligned.sum(dim=0) / torch.clamp((aligned != 0).sum(dim=0).float(), min=1.0)
    return fused


def valid(config , hypernetwork, model, tok, valid_loader, retriever, retriever_tok, decomposer, decomposer_tok):
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
        use_delta_K, use_delta_V = None, None
        ixx = 0
        try:
            while True:
                subquestion, prompt = question_decomposition(initial_prompt, decomposer, decomposer_tok)
                if not subquestion:
                    base_input = initial_prompt.strip() + '\nQuestion: {}\nAnswer:'.format(question)
                    base_input = base_input[split_index:]
                    base_input_token = {k: v.cuda() for k, v in tok(base_input, return_tensors="pt").items()}
                    passage_input_token = {k: v.cuda() for k, v in tok(initial_prompt.strip()[split_index:], return_tensors="pt").items()}
                    input_embeds = model.model.embed_tokens(passage_input_token['input_ids'])
                    delta_K, delta_V = hypernetwork(input_embeds)
                    use_delta_K = fuse_weights_batch(use_delta_K, delta_K)
                    use_delta_V = fuse_weights_batch(use_delta_V, delta_V)
                    inference_hook = target_layer.register_forward_hook(make_simple_cross_attn_hook(use_delta_K, use_delta_V))
                    final_answer = model_generation_subanswer(base_input, base_input_token, model, tok)
                    inference_hook.remove()
                    break

                fact_ids = retrieve_facts(subquestion, embs, retriever, retriever_tok, k=10)

                for i in range(len(fact_ids)):
                    fact = facts[fact_ids[i]]
                    tok_fact = {k: v.cuda() for k, v in tok(fact, return_tensors="pt").items()}
                    input_embeds = model.model.embed_tokens(tok_fact['input_ids'])
                    delta_K, delta_V = hypernetwork(input_embeds)

                    if i==0:
                        fact_delta_K = delta_K
                        fact_delta_V = delta_V
                    else:
                        fact_delta_K = mean_fuse(fact_delta_K, delta_K)
                        fact_delta_V = mean_fuse(fact_delta_V, delta_V)
                if ixx == 0:
                    use_delta_K = fact_delta_K
                    use_delta_V = fact_delta_V
                else:
                    use_delta_K = fuse_weights_batch(use_delta_K, fact_delta_K)
                    use_delta_V = fuse_weights_batch(use_delta_V, fact_delta_V)
                target_layer = model.model.layers[config.single_layer]
                inference_hook = target_layer.register_forward_hook(make_simple_cross_attn_hook(use_delta_K, use_delta_V))
                # base_input = 'Question: {}\nAnswer:'.format(subquestion)
                base_input = 'Passage: {}\nQuestion: {}\nAnswer:'.format('. '.join([facts[fact_ids[i]] for i in range(len(fact_ids))]), subquestion)
                base_input_token = {k: v.cuda() for k, v in tok(base_input, return_tensors="pt").items()}

                sub_answer = model_generation_subanswer(base_input, base_input_token, model, tok)
                print(sub_answer)
                inference_hook.remove()
                initial_prompt = prompt + '\n' + 'Sub-answer: {}\n'.format(sub_answer)
                ixx+=1
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

    hypernetwork = HyperKVGeneratorFixed(input_dim=config.embed_feature, hidden_dim=config.hid_feature,
                                         d_model=config.embed_feature,
                                         num_kv=config.num_kv).cuda()

    state_dict = torch.load(config.hypernetwork_ckpt.format(config.model_name.split("/")[-1]+'_'+config.data_name,
                                                            config.single_layer,
                                                            config.num_kv))
    hypernetwork.load_state_dict(state_dict)

    model, tok = make_main_model(config)

    valid_loader = make_Validation_loader(config, tok)
    hypernetwork.eval()
    model.eval()
    metrics = valid(config , hypernetwork, model, tok, valid_loader, retriever, retriever_tok, decomposer, decomposer_tok)


if __name__ == '__main__':
    main()