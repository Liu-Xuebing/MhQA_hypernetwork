from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
import torch
from prepare_data_for_decomposition import make_Training_loader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F


class Config:
    def __init__(self):
        self.train_batch_size = 16
        self.MQuAKE_train_dataset = '/disk/liuxb/code/Multi-EMoE/datasets/MQuAKE_train.json'
        self.data_name = 'MQuAKE'
        self.model_name = "tiiuae/Falcon3-1B-Base"



def cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
):
    if len(logits.shape) == 2:
        return F.binary_cross_entropy_with_logits(logits, labels)
    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        return F.cross_entropy(logits, labels)


config = Config()
# 加载模型
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map="auto")

# 数据加载器
train_loader = make_Training_loader(config, tokenizer)

optimizer = AdamW(model.parameters(), lr=1e-5)
# T_max = 总步数 或 每个 epoch 的 step 数
num_training_steps = len(train_loader) * 1   # 10 epoch
scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)

model.train()
for epoch in range(1):
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch).logits
        loss = cross_entropy(outputs, batch["labels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()   # 🔹 每 step 更新学习率

        epoch_loss += loss.item()

    lr_now = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1} avg loss: {epoch_loss / len(train_loader):.4f}, lr={lr_now:.6f}")

    save_dir = "./falcon3-1b-mquake"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)