import torch
import torch.nn as nn
import math

# 简单的 Transformer-style 去噪器
class SimpleTransformerDenoiser(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, t_embed):
        # 加入时间步嵌入作为条件（广播到序列长度）
        x = x + t_embed.unsqueeze(1)  # [B, 1, D]
        return self.transformer(x)


# 时间步编码（类似Positional Encoding）
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb  # [B, D]


# DiffusionExpert：用于语言模型中间层表示的扩散专家
class DiffusionExpert(nn.Module):
    def __init__(self, dim, steps=10):
        super().__init__()
        self.steps = steps
        self.dim = dim
        # 使用固定的线性β调度
        self.register_buffer("betas", torch.linspace(1e-4, 0.02, steps))
        # 时间嵌入模块
        self.time_embed = SinusoidalTimeEmbedding(dim)
        # 去噪器（可替换为更复杂模型）
        self.denoiser = SimpleTransformerDenoiser(dim)

    def q_sample(self, x, noise, t):
        """
        x: [B, L, D] - 原始表示
        noise: 同形状噪声
        t: [B] - 时间步
        """
        beta_t = self.betas[t].view(-1, 1, 1)  # [B, 1, 1]
        return torch.sqrt(1 - beta_t) * x + torch.sqrt(beta_t) * noise

    def forward(self, x, step=None):
        """
        x: [B, L, D] - 语言模型中间层表示（如第11层输出）
        step: 指定扩散步（用于推理时固定）
        """
        B, L, D = x.shape
        device = x.device

        # 1. 采样扩散步 t ∈ [0, steps-1]
        if step is None:
            t = torch.randint(0, self.steps, (B,), device=device)
        else:
            t = torch.full((B,), step, device=device, dtype=torch.long)

        # 2. 添加噪声
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, noise, t)

        # 3. 获取时间嵌入
        t_embed = self.time_embed(t)  # [B, D]

        # 4. 去噪
        x_denoised = self.denoiser(x_noisy, t_embed)

        return x_denoised


if __name__ == '__main__':
    diffusion_expert = DiffusionExpert(dim=1024, steps=3)
