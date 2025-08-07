from tqdm import tqdm
from data.base import make_Training_loader
from utils import cross_entropy
from MOE_model.make_model import make_model, replace_layer
from MOE_model.diffusion_expert import DiffusionExpert
import hydra
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch



def create_pre_hook_fn(id, weights, delta):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights, delta)
    return pre_hook_fn



def train(config, hypernetwork, model, train_loader, optimizer, scheduler):
    hypernetwork.train()
    model.eval()
    running_loss = 0.0
    for ix, tuples in enumerate(tqdm(train_loader)):
        tok_tuples, tok_sentence = tuples
        input_embeds = model.model.embed_tokens(tok_sentence[0]['input_ids']) # shape(batchsize, length, embedding_dim:4096)
        delta = hypernetwork(input_embeds)
        train_hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0,[1], delta))
        wo_logits = model(**tok_tuples)["logits"]
        FT_loss = cross_entropy(wo_logits, tok_tuples["labels"])
        train_hook.remove()
        optimizer.zero_grad()
        FT_loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += FT_loss.item()
        if ix%200 == 199:
            print(f"Training Loss: {(running_loss/200):.4f}")
            running_loss = 0.0



@hydra.main(config_path="config", config_name="config")
def main(config):
    hypernetwork = DiffusionExpert(config.embed_feature, ).cuda()

    model, tok = make_model(config)
    original_layer = model.model.layers[config.single_layer].mlp

    replace_layer(config, model, original_layer, num_experts=config.num_experts)

    for name, param in model.named_parameters():
        param.requires_grad = False

    print('training logging: {}, {}, {}'.format(config.single_layer, config.data_name, config.model_name))

    optimizer = AdamW(hypernetwork.parameters(), lr=config.learning_rate)
    train_loader = make_Training_loader(config, tok)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=config.learning_rate_min)
    train(config, hypernetwork, model, train_loader, optimizer, scheduler)

    torch.save(hypernetwork.state_dict(), config.hypernetwork_ckpt.format(config.model_name.split("/")[-1], config.data_name, config.single_layer))


if __name__ == '__main__':
    main()