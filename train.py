import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import time
import torch.nn.functional as F
from utils import get_sent_embeddings, retrieve_facts, get_word
from tqdm import tqdm
from data.base import make_Training_loader
from utils import cross_entropy
from MOE_model.make_model import make_main_model, replace_layer
from MOE_model.hypernetwork import EnhancedHyperNetwork
import hydra
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch


# clf_model_id = "microsoft/deberta-v3-base"
# clf_tokenizer = AutoTokenizer.from_pretrained(clf_model_id)
# clf_model = AutoModelForSequenceClassification.from_pretrained(clf_model_id, num_labels=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# clf_model.to(device)

def create_pre_hook_fn(id, weights, delta):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights, delta)
    return pre_hook_fn

def param_flatten(delta):
    delta_params = torch.cat([delta[i].flatten(start_dim=0) for i in range(len(delta))], dim=0)  # shape: [total_params]
    return delta_params

def train(config, hypernetwork, model, tok, train_loader, optimizer, scheduler):
    hypernetwork.train()
    model.eval()
    running_loss = []
    for ix, tuples in enumerate(tqdm(train_loader)):
        tok_tuples, tok_sentence, question, answer, passage = tuples
        input_embeds = model.model.embed_tokens(tok_sentence[0]['input_ids'].cuda()) # shape(batchsize, length, embedding_dim:4096)
        for layer_index in config.single_layer:
            delta = hypernetwork(input_embeds)
            train_hook = model.model.layers[layer_index].mlp.register_forward_pre_hook(create_pre_hook_fn(0,[1], param_flatten(delta)))
        wo_logits = model(**tok_tuples)["logits"]
        FT_loss = cross_entropy(wo_logits, tok_tuples["labels"])
        train_hook.remove()

        # pre_dict_word = get_word(wo_logits, tok_tuples["labels"], tok)
        # label = 1 if pre_dict_word.strip() == answer.strip() else 0
        # label_tensor = torch.tensor([label], dtype=torch.long, device=model.device)
        # clf_text = f"Passage: {passage}\nQuestion: {question}\nAnswer: {pre_dict_word.strip()}"
        # clf_enc = clf_tokenizer(clf_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
        # clf_out = clf_model(**clf_enc)
        # clf_loss = F.cross_entropy(clf_out.logits, label_tensor)

        loss = FT_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss.append(loss.item())
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
    state_dict = torch.load('/disk/liuxb/code/Multi-EMoE/Llama-3.1-8B_pretraining_6_hypernetwork.pth')
    hypernetwork.load_state_dict(state_dict)
    model, tok = make_main_model(config)

    for layer_index in config.single_layer:
        original_layer = model.model.layers[layer_index].mlp
        replace_layer(config, model, original_layer, layer_index)

    for name, param in model.named_parameters():
        param.requires_grad = False

    train_loader = make_Training_loader(config, tok, samples=None)

    print('pre-training logging: model: {}, layers: {}, datas_length: {}'.format(config.model_name, config.single_layer, len(train_loader)))

    optimizer = AdamW([
        {"params": hypernetwork.parameters(), "lr": config.learning_rate},
        # {"params": clf_model.parameters(), "lr": config.deberta_lr},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=config.learning_rate_min)
    train(config, hypernetwork, model, tok, train_loader, optimizer, scheduler)
    torch.save(hypernetwork.state_dict(),
               config.hypernetwork_ckpt.format(config.model_name.split("/")[-1]+'_'+config.data_name,
                                               ','.join([str(layer_index) for layer_index in config.single_layer])))


if __name__ == '__main__':
    main()