import os.path

from experiments.causal_trace import ModelAndTokenizer
from experiments.causal_trace import (make_inputs,
                                      predict_from_input,
                                      decode_tokens,
                                      find_token_range,
                                      layername,
                                      find_token_range_by_ids,
                                      plot_trace_heatmap,
                                      collect_embedding_std)
import numpy
from collections import defaultdict

from util import nethook
import torch
import json
from dsets import CTDataset
from tqdm import tqdm


model_name = "meta-llama/Llama-3.1-8B"
mt = ModelAndTokenizer(
    model_name,
    torch_dtype=(torch.float16 if "20b" in model_name else None),)


def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    trace_layers=None,  # List of traced outputs to return
):
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")


    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers

    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)
    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced
    return probs



def trace_important_states(model, num_layers, inp, e_range, answer_t):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for layer in range(num_layers):
        r = trace_with_patch(
                model,
                inp,
                [(list(range(e_range[0], e_range[1])), layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
            )
        table.append(r)
    return torch.stack(table)




def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=1):
    ntoks = inp['input_ids'].shape[1]
    table = []
    for layer in range(num_layers):
        layerlist = [(list(range(e_range[0], e_range[1])), layername(model, L, kind))
                     for L in range( max(0, layer - window // 2),
                                     min(num_layers, layer - (-window // 2)))]

        r = trace_with_patch(model, inp, layerlist, answer_t, tokens_to_mix=e_range)
        table.append(r)
    return torch.stack(table)


def calculate_hidden_flow(
    mt, text_prompt, question_prompt, answer, window=1, kind=None):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [text_prompt, question_prompt])
    answer_t = mt.tokenizer(' {}'.format(answer), return_tensors='pt',add_special_tokens=False,)['input_ids'][0][0]
    with torch.no_grad():
        # print(predict_from_input(mt.model, answer_t, inp))
        base_score, low_score = predict_from_input(mt.model, answer_t, inp)
        base_score = base_score.cpu().numpy()
        low_score = low_score.cpu().numpy()
        # print(base_score, low_score)
    # [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range_by_ids(inp['input_ids'][0], inp['input_ids'][1], mt.tokenizer.eos_token_id)
    print(e_range)
    # low_score = trace_with_patch(
    #     mt.model, inp, [], answer_t, e_range).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t)
    else:
        differences = trace_important_window(
            mt.model, mt.num_layers, inp, e_range, answer_t, window=window, kind=kind)

    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp,
        input_tokens=decode_tokens(mt.tokenizer, inp['input_ids'][0]),
        subject_range=e_range,
        # answer=answer,
        window=window,
        kind=kind or "")



def plot_hidden_flow(
    mt,
    text_prompt,
    question_prompt,
    answer,
    window=1,
    kind=None,
    savepdf=None,
    indexi=None,):

    result = calculate_hidden_flow(
        mt, text_prompt, question_prompt, answer, window=window, kind=kind)
    with open("{}/{}_{}.json".format(savepdf, kind, indexi), "without Hypernetwork") as f:
        json.dump([{"scores": result['scores'].tolist(),
                        "high_score": result['high_score'].tolist(),
                        "low_score": result['low_score'].tolist()}], f, ensure_ascii=False, indent=4)
    # plot_trace_heatmap(result, savepdf, modelname=modelname)



def plot_all_flow(mt, text_prompt, question_prompt=None, answer=None, modelname=None, indexi=None):
    for kind in ["self_attn", "mlp"]:
        plot_hidden_flow(
            mt, text_prompt, question_prompt, answer, kind=kind, savepdf='./results', indexi=indexi)

# plot_all_flow(mt, "Query: When did richmond last play in a preliminary final?\nAnswer:", noise=noise_level, modelname=model_name.split('/')[-1])


knowns = CTDataset(data_dir='/disk/liuxb/code/Multi-EMoE/datasets/causal_tracing.json')  # Dataset of known facts

for ix, knowledge in enumerate(tqdm(knowns)):
    if os.path.exists("{}/{}_{}.json".format('./results', 'mlp', ix)):
        continue
    plot_all_flow(mt=mt,
                  text_prompt='{}\nQuestion: {}\nAnswer:'.format(knowledge['text'], knowledge["question"]),
                  question_prompt="Question: {}\nAnswer:".format(knowledge["question"]),
                  answer = knowledge['answer'],
                  modelname=model_name.split('/')[-1], indexi = ix)

