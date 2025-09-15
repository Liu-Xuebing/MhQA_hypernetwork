from tqdm import tqdm
from data.base import make_Training_loader
from utils import cross_entropy
from MOE_model.make_model import make_model, replace_layer
from MOE_model.hypernetwork import EnhancedHyperNetwork, HyperNetwork
import hydra
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import time


def create_pre_hook_fn(id, weights, delta):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights, delta)
    return pre_hook_fn



def train(config, hypernetwork, model, train_loader, optimizer, scheduler):
    hypernetwork.train()
    model.eval()
    running_loss = []
    for ix, tuples in enumerate(tqdm(train_loader)):
        tok_tuples, tok_sentence = tuples
        input_embeds = model.model.embed_tokens(tok_sentence[0]['input_ids']) # shape(batchsize, length, embedding_dim:4096)
        for layer_index in config.single_layer:
            delta = hypernetwork(input_embeds)
            train_hook = model.model.layers[layer_index].mlp.register_forward_pre_hook(create_pre_hook_fn(0,[1], delta))
        wo_logits = model(**tok_tuples)["logits"]
        FT_loss = cross_entropy(wo_logits, tok_tuples["labels"])
        train_hook.remove()
        optimizer.zero_grad()
        FT_loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss.append(FT_loss.item())
        if (ix+1)%49 == 0:
            print(f"Training Loss: {sum(running_loss)/len(running_loss):.4f}")
            running_loss = []


@hydra.main(config_path="config", config_name="config")
def main(config):
    hypernetwork = EnhancedHyperNetwork(embed_dim = config.embed_feature,
                                        rank = config.rank,
                                        input_dim = config.in_feature,
                                        hidden_dim = config.hid_feature,
                                        output_dim = config.out_feature).cuda()

    model, tok = make_model(config)

    for layer_index in config.single_layer:
        original_layer = model.model.layers[layer_index].mlp
        replace_layer(config, model, original_layer, layer_index)

    for name, param in model.named_parameters():
        param.requires_grad = False

    train_loader = make_Training_loader(config, tok, samples=None)
    #
    print('pre-training logging: model: {}, layers: {}, datas_length: {}'.format(config.model_name, config.single_layer, len(train_loader)))
    #
    optimizer = AdamW(hypernetwork.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=config.learning_rate_min)
    train(config, hypernetwork, model, train_loader, optimizer, scheduler)
    torch.save(hypernetwork.state_dict(),
               config.hypernetwork_ckpt.format(config.model_name.split("/")[-1],
                                               ','.join([str(layer_index) for layer_index in config.single_layer])))


if __name__ == '__main__':
    main()