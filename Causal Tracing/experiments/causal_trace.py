import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
import torch
from matplotlib import pyplot as plt
import os

class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """
    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, trust_remote_code=True
            )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(model)\.(layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level



def make_inputs(tokenizer, prompts, device="cuda"):
    # token_lists = [tokenizer.encode(p) for p in prompts]
    # maxlen = max(len(t) for t in token_lists)
    # if "[PAD]" in tokenizer.all_special_tokens:
    #     pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    # else:
    #     pad_id = tokenizer.pad_token_id
    # input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    # # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    # attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    # return dict(
    #     input_ids=torch.tensor(input_ids).to(device),
    #     #    position_ids=torch.tensor(position_ids).to(device),
    #     attention_mask=torch.tensor(attention_mask).to(device),
    # )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    token_list = tokenizer(prompts, return_tensors='pt',add_special_tokens=False, padding=True)
    token_list = {k: v.to(device) for k, v in token_list.items()}
    # extra_token = torch.full((len(prompts), 1), 126336, device=device, dtype=input_ids.dtype)
    # input_ids = torch.cat([input_ids, extra_token], dim=1)
    return token_list


def predict_from_input(model, answer_t, inp):
    out = model(**inp)["logits"]
    probs = torch.softmax(out[:, -1], dim=1)
    answer_probs = probs[:, answer_t]
    return answer_probs[0], answer_probs[1]



def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode(t, skip_special_tokens=True) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_loc = whole_string.index(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)


def find_token_range_by_ids(input_ids_full, input_ids_sub, pad_token_id=None):
    """
    找到 input_ids_sub 中非 pad token 在 input_ids_full 的连续位置。
    返回 (start, end)
    """
    # 去掉 padding
    if pad_token_id is not None:
        input_ids_sub = input_ids_sub[input_ids_sub != pad_token_id]

    full = input_ids_full.tolist()
    sub = input_ids_sub.tolist()
    n, m = len(full), len(sub)

    for i in range(n - m + 1):
        if full[i:i + m] == sub:
            return (i, i + m)

    raise ValueError("substring not found")


def layername(model, num, kind=None):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        if kind == "embed":
            return "model.embed_tokens"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model") and hasattr(model.model, "transformer"): #DLM
        if kind == "embed":
            return "model.transformer.wte"
        return f"model.transformer.blocks.{num}{'' if kind is None else '.' + kind}"
    assert False, "unknown transformer structure"


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    if differences.dim() == 1:
        differences = differences.unsqueeze(0)
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):
        labels[i] = labels[i] + "*"

    with plt.rc_context(rc={"font.family": "DejaVu Serif"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "self_attn": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(["Sentence"])
        if not kind:
            kindname = "block"
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "Attn" if kind == "self_attn" else "MLP"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig("{}/{}.png".format(savepdf, kindname), bbox_inches="tight")
            plt.close()
        else:
            plt.show()