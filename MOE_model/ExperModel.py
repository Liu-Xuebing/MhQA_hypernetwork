import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)


class LowRankExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank):
        """
        Expert module with one hidden layer, ReLU activation, and low-rank decomposition.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of the hidden layer.
            output_dim (int): Dimensionality of the output features.
            rank (int): Rank for low-rank decomposition.
        """
        super(LowRankExpert, self).__init__()

        # Low-rank decomposition for hidden layer
        self.hidden_layer_u = nn.Linear(input_dim, rank, bias=False)  # First projection
        self.hidden_layer_v = nn.Linear(rank, hidden_dim, bias=False)  # Second projection

        self.activation = nn.ReLU()

        # Low-rank decomposition for output layer
        self.output_layer_u = nn.Linear(hidden_dim, rank, bias=False)  # First projection
        self.output_layer_v = nn.Linear(rank, output_dim, bias=False)  # Second projection

    def forward(self, x):
        """
        Forward pass through the low-rank expert module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Low-rank hidden layer
        x = self.hidden_layer_u(x)  # Project to low-rank space
        x = self.hidden_layer_v(x)  # Project to hidden_dim space
        x = self.activation(x)

        # Low-rank output layer
        x = self.output_layer_u(x)  # Project to low-rank space
        x = self.output_layer_v(x)  # Project to output_dim space
        return x



class MoE(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, output_dim=4096, rank=512, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.expert = LowRankExpert(input_dim, hidden_dim, output_dim, rank=rank).cuda()
        self.experts = nn.ModuleList(self.expert for _ in range(num_experts))


    def forward(self, x, scores, delta):
        if delta is None:
            output = torch.zeros(x.shape, device=x.device)
            if isinstance(scores, list):
                scores = torch.Tensor(scores).to(x.device)
            else:
                scores = scores.to(x.device)
            for i, expert in enumerate(self.experts):
                expert_output = expert(x) * scores[i]
                output += expert_output
            return output
        else:
            # if self.num_experts == 1:
            #     r, in_d = self.expert.hidden_layer_u.weight.shape
            #     h_d, _ = self.expert.hidden_layer_v.weight.shape
            #     r2, h2 = self.expert.output_layer_u.weight.shape
            #     o_d, _ = self.expert.output_layer_v.weight.shape
            #     i = 0
            #     d1 = delta[:, i:i + r * in_d].view(r, in_d)
            #     i += r * in_d
            #     d2 = delta[:, i:i + h_d * r].view(h_d, r)
            #     i += h_d * r
            #     d3 = delta[:, i:i + r2 * h2].view(r2, h2)
            #     i += r2 * h2
            #     d4 = delta[:, i:i + o_d * r].view(o_d, r)
            #     W1 = self.expert.hidden_layer_u.weight + d1.to(self.expert.hidden_layer_u.weight.device)
            #     W2 = self.expert.hidden_layer_v.weight + d2.to(self.expert.hidden_layer_v.weight.device)
            #     W3 = self.expert.output_layer_u.weight + d3.to(self.expert.output_layer_u.weight.device)
            #     W4 = self.expert.output_layer_v.weight + d4.to(self.expert.output_layer_v.weight.device)
            #     x = F.linear(x, W1)
            #     x = F.linear(x, W2)
            #     x = self.expert.activation(x)
            #     x = F.linear(x, W3)
            #     output = F.linear(x, W4)
            # else:
            output = torch.zeros(x.shape, device=x.device)
            if isinstance(scores, list):
                scores = torch.Tensor(scores).to(x.device)
            else:
                scores = scores.to(x.device)

            # 单个专家参数大小
            r, in_d = self.experts[0].hidden_layer_u.weight.shape
            h_d, _ = self.experts[0].hidden_layer_v.weight.shape
            r2, h2 = self.experts[0].output_layer_u.weight.shape
            o_d, _ = self.experts[0].output_layer_v.weight.shape


            for i, expert in enumerate(self.experts):
                d = delta[i]
                j = 0
                d1 = d[:, j:j + r * in_d].view(-1, r, in_d)
                j += r * in_d
                d2 = d[:, j:j + h_d * r].view(-1, h_d, r)
                j += h_d * r
                d3 = d[:, j:j + r2 * h2].view(-1, r2, h2)
                j += r2 * h2
                d4 = d[:, j:j + o_d * r].view(-1, o_d, r)

                W1 = expert.hidden_layer_u.weight.unsqueeze(0) + d1.to(x.device)
                W2 = expert.hidden_layer_v.weight.unsqueeze(0) + d2.to(x.device)
                W3 = expert.output_layer_u.weight.unsqueeze(0) + d3.to(x.device)
                W4 = expert.output_layer_v.weight.unsqueeze(0) + d4.to(x.device)

                x_proj = F.linear(x, W1.squeeze(0))
                x_proj = F.linear(x_proj, W2.squeeze(0))
                x_proj = expert.activation(x_proj)
                x_proj = F.linear(x_proj, W3.squeeze(0))
                x_proj = F.linear(x_proj, W4.squeeze(0))

                output += x_proj * scores[i]
            return output



# 定义并行处理的函数
class ParallelFFNMoE(nn.Module):
    def __init__(self, ffn, moes):
        super(ParallelFFNMoE, self).__init__()
        self.ffn = ffn
        self.moes = moes

    def forward(self, x, step=None):
        out_main  = self.ffn(x)
        out_denoised = self.moes(x, step)
        outputs = out_main + out_denoised

        return outputs, out_denoised, x