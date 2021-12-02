import torch
from torch import nn, Tensor
import math


class PositionalEncoding(nn.Module):
    """
    from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    """
    inspired by https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    """

    def __init__(self, input_size, hidden_size, n_heads, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        # self.W_Q = nn.Linear(input_size, hidden_size*n_heads)
        # self.W_K = nn.Linear(input_size, hidden_size*n_heads)
        # self.W_V = nn.Linear(input_size, hidden_size*n_heads)
        self.W_QKV = nn.Linear(input_size, 3 * (hidden_size * n_heads))
        self.W_O = nn.Linear(hidden_size * n_heads, hidden_size)
        self.Dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.W_QKV.weight)
        nn.init.xavier_uniform_(self.W_O.weight)

    def forward(self, x, mask=None):
        batch_size, seq_length = x.shape[0], x.shape[1]
        # q = self.W_Q(x)
        # k = self.W_K(x)
        # v = self.W_V(x)
        qkv = self.W_QKV(x)
        # Separate Q, K, V from linear output
        # [Batch, Head, SeqLen, Dims]
        qkv = qkv.view(batch_size, seq_length, self.n_heads, 3 * self.hidden_size).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.input_size)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e10)
        attention = attn_logits.softmax(dim=-1)
        v = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, seq_length, -1)
        o = self.W_O(v)

        return self.Dropout(o)


class MyConv1d(nn.Module):

    def __init__(self, input_size, hidden_size, kernel):
        super(MyConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel, padding=kernel // 2)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class FFTBlock(nn.Module):

    def __init__(self, hidden_size, n_heads, conv_size, kernel, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.attn = MultiHeadedAttention(hidden_size, hidden_size, n_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.conv = nn.Sequential(
            MyConv1d(hidden_size, conv_size, kernel),
            nn.ReLU(inplace=True),
            MyConv1d(conv_size, hidden_size, kernel),
            nn.Dropout(p=dropout)
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        residual = x
        x = self.attn(x, mask)
        x = self.norm1(x)
        x += residual
        residual = x
        x = self.conv(x)
        x = self.norm2(x)
        x += residual
        return x


class LengthRegulator(nn.Module):
    """
    inspired by https://github.com/xcmyz/FastSpeech/blob/master/modules.py
    """

    def __init__(self, input_size, hidden_size, kernel, dropout=0.1):
        super(LengthRegulator, self).__init__()
        self.impl = nn.Sequential(
            MyConv1d(input_size, hidden_size, kernel),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            MyConv1d(hidden_size, hidden_size, kernel),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x, true_duration: Tensor = None):
        # [batch_size, seq_len, embedding_dim]
        pred_duration = self.impl(x).squeeze(-1)
        if true_duration is None:
            use_duration = torch.exp(pred_duration)
        else:
            use_duration = true_duration
        use_duration = use_duration.round().int()
        length = int(torch.max(torch.sum(use_duration, dim=-1), dim=-1)[0])
        alignment = torch.zeros(size=(x.shape[0], x.shape[1], length), device=x.device)
        alignment = get_alignment(alignment, use_duration)

        x = x.transpose(-1, -2) @ alignment
        return x.transpose(-1, -2), pred_duration


@torch.no_grad()
def get_alignment(alignment, duration):
    batch, length = duration.shape
    for i in range(batch):
        count = 0
        for j in range(length):
            alignment[i][j][count:count + duration[i][j]] = 1
            count += duration[i][j]
    return alignment


class FastSpeech(nn.Module):

    def __init__(self, vocab_size, n_ph_block=6, n_melspec_block=6, hidden_size=384, n_attn_heads=2, kernel=3,
                 conv_size=1536, lin_size=80, dropout=0.1, max_ph_len=1000, max_melspec_len=1000):
        super(FastSpeech, self).__init__()

        self.ph_embedding = nn.Embedding(vocab_size, hidden_size)
        self.ph_PE = PositionalEncoding(hidden_size, max_len=max_ph_len)
        self.ph_FFTBlocks = nn.Sequential(*[
            FFTBlock(hidden_size, n_attn_heads, conv_size, kernel, dropout) for i in range(n_ph_block)
        ])
        self.length_reg = LengthRegulator(hidden_size, hidden_size, kernel, dropout)
        self.melspec_PE = PositionalEncoding(hidden_size, max_len=max_melspec_len)
        self.melspec_FFTBlocks = nn.Sequential(*[
            FFTBlock(hidden_size, n_attn_heads, conv_size, kernel, dropout) for i in range(n_melspec_block)
        ])
        self.linear = nn.Linear(hidden_size, lin_size)

    def forward(self, x, true_duration):
        x = self.ph_embedding(x)
        # [batch_size, seq_len, embedding_dim]
        x = self.ph_PE(x.transpose(0, 1)).transpose(0, 1)
        x = self.ph_FFTBlocks(x)
        x, durations = self.length_reg(x, true_duration)
        x = self.melspec_PE(x.transpose(0, 1)).transpose(0, 1)
        x = self.melspec_FFTBlocks(x)
        x = self.linear(x)
        return x.transpose(1, 2), durations
