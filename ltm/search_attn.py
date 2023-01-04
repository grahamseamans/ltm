# https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MultiHeadAttention", "ScaledDotProductAttention"]


class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttentionSearch(nn.Module):
    def __init__(
        self,
        num_heads,
        embedding_dim,
        q_features,
        mem_features,
        bias=True,
        activation=F.relu,
    ):
        """Multi-head attention.
        :param mem_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionSearch, self).__init__()
        # if mem_features % head_num != 0:
        #     raise ValueError('`mem_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.mem_features = mem_features
        self.num_heads = num_heads
        self.activation = activation
        self.bias = bias
        self.embedding_dim = embedding_dim
        self.linear_q = nn.Linear(q_features, embedding_dim * num_heads, bias)
        self.linear_k = nn.Linear(mem_features, embedding_dim * num_heads, bias)
        self.linear_v = nn.Linear(mem_features, embedding_dim * num_heads, bias)
        # self.linear_o = nn.Linear(mem_features, in_features * head_num, bias)

    def forward(self, q, k, v, mask=None):
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # print()

        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # print()

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)

        # print(q.shape)
        # print(k.shape)
        # print(v.shape)
        # print()

        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        # print(y.shape)
        y = self._reshape_from_batches(y)
        # print(y.shape)

        # y = self.linear_o(y)
        # if self.activation is not None:
        #     y = self.activation(y)
        # print(y.shape)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return (
            torch.tril(torch.ones(seq_len, seq_len))
            .view(1, seq_len, seq_len)
            .repeat(batch_size, 1, 1)
        )

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.num_heads
        return (
            x.reshape(batch_size, seq_len, self.num_heads, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        x = x.reshape(batch_size, self.num_heads * seq_len, in_feature)
        return x

    def extra_repr(self):
        return "mem_features={}, head_num={}, bias={}, activation={}".format(
            self.mem_features,
            self.num_heads,
            self.bias,
            self.activation,
        )
