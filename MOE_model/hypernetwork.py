import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EnhancedHyperNetwork(nn.Module):
    def __init__(self,
                 embed_dim: int,         # 输入知识向量维度
                 rank: int,              # expert 的低秩维度
                 input_dim: int,         # expert 第一层输入维度
                 hidden_dim: int,        # expert 隐层维度
                 output_dim: int,        # expert 输出维度
                 num_heads: int = 4,     # 多头注意力个数
                 hidden_size: int = 256, # hypernet 隐层宽度
                 num_basis: int = 8      # basis 个数（建议 4~16）
    ):

        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_basis = num_basis

        # --------- 1. Knowledge Encoder (Self-Attention) ---------
        self.attn_proj = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)

        # --------- 2. Feedforward HyperNetwork Core ---------
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # --------- 3. Coefficient Generators (for each part) ---------
        # 输出的是系数而非完整参数，参数将基于 basis 加权组合
        self.coeff_u1 = nn.Linear(hidden_size, num_basis)
        self.coeff_v1 = nn.Linear(hidden_size, num_basis)
        self.coeff_u2 = nn.Linear(hidden_size, num_basis)
        self.coeff_v2 = nn.Linear(hidden_size, num_basis)

        # --------- 4. Basis Parameters (共享矩阵) ---------
        self.basis_u1 = nn.Parameter(torch.randn(num_basis, rank, input_dim))     # [B, r, d]
        self.basis_v1 = nn.Parameter(torch.randn(num_basis, hidden_dim, rank))    # [B, d', r]
        self.basis_u2 = nn.Parameter(torch.randn(num_basis, rank, hidden_dim))
        self.basis_v2 = nn.Parameter(torch.randn(num_basis, output_dim, rank))

        self._init_weights()

    def _init_weights(self):
        # 初始化 basis 更稳定
        nn.init.xavier_uniform_(self.basis_u1)
        nn.init.xavier_uniform_(self.basis_v1)
        nn.init.xavier_uniform_(self.basis_u2)
        nn.init.xavier_uniform_(self.basis_v2)

    def forward(self, x_embed):  # x_embed: [B, T, D]
        """
        x_embed: 外部知识的 token embedding（batch_size, seq_len, embed_dim）
        输出：拼接后的专家参数向量 [B, Total_Params]
        """
        # --------- Step 1: Encode Knowledge ---------
        attn_output, _ = self.attn_proj(x_embed, x_embed, x_embed)  # shape: [B, T, D]
        x_encoded = self.attn_norm(attn_output + x_embed)
        pooled = x_encoded.mean(dim=1)  # shape: [B, D]

        # --------- Step 2: Feedforward Layers ---------
        h = self.ffn(pooled)  # shape: [B, H]

        # --------- Step 3: Generate Coefficients for Each Part ---------
        alpha_u1 = F.softmax(self.coeff_u1(h), dim=-1)  # shape: [B, K]
        alpha_v1 = F.softmax(self.coeff_v1(h), dim=-1)
        alpha_u2 = F.softmax(self.coeff_u2(h), dim=-1)
        alpha_v2 = F.softmax(self.coeff_v2(h), dim=-1)

        # --------- Step 4: Weighted Combination from Basis ---------
        # u1 = Σ α_k * U_k → shape: [B, r, d]
        u1 = torch.einsum('bk,krd->brd', alpha_u1, self.basis_u1)
        v1 = torch.einsum('bk,kdr->bdr', alpha_v1, self.basis_v1)
        u2 = torch.einsum('bk,krd->brd', alpha_u2, self.basis_u2)
        v2 = torch.einsum('bk,kdr->bdr', alpha_v2, self.basis_v2)

        # --------- Step 5: Flatten as Expert Delta Parameters ---------
        # 最终拼接为一维向量输出
        delta_params = torch.cat([
            u1.flatten(start_dim=1),
            v1.flatten(start_dim=1),
            u2.flatten(start_dim=1),
            v2.flatten(start_dim=1)
        ], dim=-1)  # shape: [B, total_params]

        return delta_params