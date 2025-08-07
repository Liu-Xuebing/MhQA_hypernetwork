from tqdm import tqdm
from data.base import make_Training_loader
from utils import cross_entropy, set_seed, print_trainable_parameters
from MOE_model.make_model import make_model, replace_layer
import hydra
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch



def train(config, model, train_loader, optimizer, scheduler):
    model.eval()
    running_loss = 0.0
    for ix, tuples in enumerate(tqdm(train_loader)):
        tok_tuples, tok_sentence = tuples
        wo_logits = model(**tok_tuples)["logits"]
        FT_loss = cross_entropy(wo_logits, tok_tuples["labels"])
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
    model, tok = make_model(config)

    if "Llama" in config.model_name:
        original_layer = model.model.layers[config.single_layer].mlp
    elif 'gpt' in config.model_name:
        original_layer = model.transformer.h[config.single_layer].mlp
    else:
        raise AssertionError

    replace_layer(config, model, original_layer)
    print_trainable_parameters(model)
    print('training logging: {}, {}, {}'.format(config.single_layer, config.data_name, config.model_name))

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    train_loader = make_Training_loader(config, tok, samples=10000)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=config.learning_rate_min)
    train(config, model, train_loader, optimizer, scheduler)

if __name__ == '__main__':
    set_seed()
    main()