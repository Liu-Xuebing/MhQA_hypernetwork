import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#
class EnhancedHyperNetwork(nn.Module):
    def __init__(self,
                 embed_dim: int,         # 输入知识向量维度
                 rank: int,              # expert 的低秩维度
                 input_dim: int,         # expert 第一层输入维度
                 hidden_dim: int,        # expert 隐层维度
                 output_dim: int,        # expert 输出维度
                 num_heads: int = 4,     # 多头注意力个数
                 hidden_size: int = 2048, # hypernet 隐层宽度
                 num_basis: int = 4,
                 dropout: int = 0.1,
                 num_layers = 2
    ):

        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_basis = num_basis

        # --------- 1. Knowledge Encoder (Self-Attention) ---------
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

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
        x_embed:  token embedding（batch_size, seq_len, embed_dim）
        """
        # --------- Step 1: Encode Knowledge ---------
        B = x_embed.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x_with_cls = torch.cat([cls_tokens, x_embed], dim=1)  # [B, 1+T, D]
        attn_output = self.transformer(x_with_cls)

        pooled = attn_output[:, 0, :]  # [B, D]
        # pooled = torch.mean(x_embed, dim=1, keepdim=True)
        # print(pooled)
        # --------- Step 2: Feedforward Layers ---------
        h = self.ffn(pooled)  # shape: [B, H]

        # --------- Step 3: Generate Coefficients for Each Part ---------
        alpha_u1 = F.softmax(self.coeff_u1(h)/2.0, dim=-1)  # shape: [B, K]
        alpha_v1 = F.softmax(self.coeff_v1(h)/2.0, dim=-1)
        alpha_u2 = F.softmax(self.coeff_u2(h)/2.0, dim=-1)
        alpha_v2 = F.softmax(self.coeff_v2(h)/2.0, dim=-1)

        print(alpha_u1)
        # --------- Step 4: Weighted Combination from Basis ---------
        # u1 = Σ α_k * U_k → shape: [B, r, d]
        u1 = torch.einsum('bk,krd->brd', alpha_u1, self.basis_u1)
        v1 = torch.einsum('bk,kdr->bdr', alpha_v1, self.basis_v1)
        u2 = torch.einsum('bk,krd->brd', alpha_u2, self.basis_u2)
        v2 = torch.einsum('bk,kdr->bdr', alpha_v2, self.basis_v2)

        # --------- Step 5: Flatten as Expert Delta Parameters ---------
        # 最终拼接为一维向量输出
        # delta_params = torch.cat([
        #     u1.flatten(start_dim=1),
        #     v1.flatten(start_dim=1),
        #     u2.flatten(start_dim=1),
        #     v2.flatten(start_dim=1)
        # ], dim=-1)  # shape: [B, total_params]
        delta_params = [u1.squeeze(), v1.squeeze(), u2.squeeze(), v2.squeeze()]
        return delta_params



class HyperKVGeneratorFixed(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model,
                 num_kv):
        super().__init__()

        self.pooling_layer = nn.Linear(input_dim, 1)
        self.att_pool = nn.Linear(input_dim, 1)

        self.num_kv = num_kv
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear_K = nn.Linear(hidden_dim, num_kv * d_model)
        self.linear_V = nn.Linear(hidden_dim, num_kv * d_model)

    def forward(self, c_emb):
        """
        c_emb: [B, S_c, embed_dim]
        return: K,V: [B, num_kv, d_model]
        """
        B, S_c, _ = c_emb.shape

        att_scores = self.att_pool(c_emb)
        # 权重: [B, S_c, 1]
        att_weights = torch.softmax(att_scores, dim=1)
        # 加权求和池化后得到 [B, input_dim]
        pooled = (c_emb * att_weights).sum(dim=1)

        hidden = self.mlp(pooled)   # [B, hidden_dim]
        K = self.linear_K(hidden).view(B, self.num_kv, -1)
        V = self.linear_V(hidden).view(B, self.num_kv, -1)

        return K, V


if __name__ == '__main__':
    pass