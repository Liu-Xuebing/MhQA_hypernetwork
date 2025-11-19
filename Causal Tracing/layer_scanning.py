import sys
sys.path.append('../')
from MOE_model.make_model import make_main_model, replace_layer
from torch.optim import AdamW
import math
import hydra
import torch
from tqdm import tqdm
from utils import cross_entropy
from MOE_model.hypernetwork import HyperKVGeneratorFixed
from MOE_model.make_model import make_main_model
from data.base import make_Training_loader
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc


def cross_attention(Q, K, V):
    """
    Q: [B, S_q, d_model]  （LM 某层输出）
    K: [B, S_k, d_model]  （超网络生成）
    V: [B, S_k, d_model]
    """
    # print(K)
    # print(V)
    d_model = Q.size(-1)
    B, S_q, _ = Q.shape
    S_k = K.shape[1]
    num_heads = 8
    d_k = d_model // num_heads

    # 多头拆分
    Qh = Q.view(B, S_q, num_heads, d_k).transpose(1, 2).cuda()  # [B, h, S_q, d_k]
    Kh = K.view(B, S_k, num_heads, d_k).transpose(1, 2).cuda()   # [B, h, S_k, d_k]
    Vh = V.view(B, S_k, num_heads, d_k).transpose(1, 2).cuda()   # [B, h, S_k, d_k]

    att = (Qh @ Kh.transpose(-2, -1)) / (d_k ** 0.5)      # [B, h, S_q, S_k]
    att = att.softmax(dim=-1)
    out = att @ Vh                                       # [B, h, S_q, d_k]

    out = out.transpose(1,2).contiguous().view(B, S_q, d_model)
    return out




def make_simple_cross_attn_hook(delta_K, delta_V):
    def hook_modify_hidden_states(module, input, output):
        # output is a tuple: (hidden_states, maybe_attn, maybe_cache)
        hidden_states = output[0]
        # Q = hidden_states
        Q = hidden_states
        # 使用你外部算好的 delta_K, delta_V进行 cross-attention
        delta = cross_attention(Q, delta_K, delta_V)
        # 修改 hidden_states
        device = hidden_states.device
        delta = delta.to(device)
        new_hidden = hidden_states + delta
        # 保持输出结构一致
        new_output = (new_hidden,) + output[1:]
        return new_output
    return hook_modify_hidden_states



def train(layer, hypernetwork, model, train_loader, optimizer, scheduler):
    all_losses = []
    # running_loss = []
    for ix, tuples in enumerate(tqdm(train_loader)):
        tok_tuples, tok_sentence = tuples
        input_embeds = model.model.embed_tokens(
            tok_sentence[0]['input_ids'].cuda())  # shape(batchsize, length, embedding_dim:4096)
        delta_K, delta_V = hypernetwork(input_embeds)

        target_layer = model.model.layers[layer]
        hook = target_layer.register_forward_hook(
            make_simple_cross_attn_hook(delta_K, delta_V)
        )
        output_logits = model(**tok_tuples)["logits"]
        FT_loss = cross_entropy(output_logits, tok_tuples["labels"])
        hook.remove()
        optimizer.zero_grad()
        FT_loss.backward()
        optimizer.step()
        scheduler.step()
        # running_loss.append(FT_loss.item())
        all_losses.append(FT_loss.item())
        # if (ix+1)%49 == 0:
        #     avg_loss = sum(running_loss) / len(running_loss)
        #     ppl = math.exp(avg_loss)
        #     print(f"Training Loss: {avg_loss:.4f}, PPL: {ppl:.2f}")
        #     running_loss = []
    all_loss = sum(all_losses) / len(all_losses)
    return math.exp(all_loss)


@hydra.main(config_path="../config", config_name="config")
def main(config):
    model, tok = make_main_model(config)
    model.eval()
    ppls = []
    for layer in list(range(32)):
        hypernetwork = HyperKVGeneratorFixed(input_dim=config.embed_feature, hidden_dim=config.hid_feature,
                                           d_model=config.embed_feature).cuda()
        other_params = [p for n, p in hypernetwork.named_parameters()]

        train_loader = make_Training_loader(config, tok, samples=None)

        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": 1e-4},  # 主干部分
        ])
        scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=config.learning_rate_min)
        hypernetwork.train()
        ppl = train(layer, hypernetwork, model, train_loader, optimizer, scheduler)
        ppls.append(ppl)
        print(ppls)
        # --- release GPU ---
        del hypernetwork
        del optimizer
        del scheduler
        del train_loader
        torch.cuda.empty_cache()  # 把不再使用的缓存显存释放
        gc.collect()  # Python 垃圾回收
    print(ppls)


if __name__ == '__main__':
    main()