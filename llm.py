import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math

def causal_mask(size):
    mask = torch.triu(torch.ones(1,size,size), diagonal=1)
    return mask==0

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)

    def forward(self, x):
        return self.embedding(x) + math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        # Create positional encodings with shape (1, seq_len, d_model)
        pe = torch.zeros(1, seq_len, d_model, dtype=torch.float32)

        positions = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        pe[0, :, 0::2] = torch.sin(positions / div_term)
        pe[0, :, 1::2] = torch.cos(positions / div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # add positional encodings for the current sequence length
        x = x + self.pe[:, :seq_len, :].detach()
        return self.dropout(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len, d_model, dropout, h):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.h = h

        assert d_model % h == 0, f"d_model {d_model} must be divisible by h {h}"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]

        # (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask==0, -1e9)

        # (To make the entire column to sum up to 1 - probability)
        attention_scores = attention_scores.softmax(dim=-1)

        attention_scores = dropout(attention_scores)

        # attention = (batch_size, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # current: (batch_size, seq_len, d_model) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, -1).transpose(1,-2)
        key = key.view(key.shape[0], key.shape[1], self.h, -1).transpose(1, -2)
        value = value.view(value.shape[0], value.shape[1], self.h, -1).transpose(1, -2)

        attention, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, h*d_k)
        attention = attention.transpose(1,2).contiguous().view(attention.shape[0], -1, self.h * self.d_k)
        return self.w_o(attention)


class FeedForward(nn.Module):
    def __init__(self, dropout, d_ff, d_model):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps = 10**-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.ones(d_model))

        self.eps = eps

    def forward(self, x):
        # x = (batch_size, seq_len, batch_size)
        mean = x.mean(dim=-1, keepdim=True) # (batch_size, seq_len, 1)

        std = x.std(dim=-1, keepdim=True) # (batch_size, seq_len, 1)

        return ((x-mean)/std+self.eps) * self.alpha + self.beta


class ResidualConnection(nn.Module):
    def __init__(self, dropout, d_model):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layernorm(x)))


class DecoderBlock(nn.Module):
    def __init__(self, multiHeadAttention, feedForward, d_model, dropout):
        super().__init__()
        self.multiHeadAttention = multiHeadAttention
        self.feedForward = feedForward
        self.residualConnection = nn.ModuleList([ResidualConnection(dropout, d_model) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residualConnection[0](x, lambda x: self.multiHeadAttention(x, x, x, mask))
        x = self.residualConnection[1](x, lambda x: self.feedForward(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layers:nn.ModuleList, d_model):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


class Transformers(nn.Module):
    def __init__(self, inputEmbedding, positionalEncoding, decoder, projection):
        super().__init__()
        self.inputEmbedding = inputEmbedding
        self.positionalEncoding = positionalEncoding
        self.decoder = decoder
        self.projection = projection


    def forward(self, x, mask=None, is_already_embedded: bool = True):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size, seq_len = x.shape[0], x.shape[1]

        if not is_already_embedded:
            # Standard Text-only path
            x = self.inputEmbedding(x)
            x = self.positionalEncoding(x)
        else:
            # For Visual Input
            pass
        # Handle Causal masking
        causal_masked = causal_mask(seq_len).to(device)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2).bool().to(device)
            full_mask = causal_masked.unsqueeze(0) & mask
        else:
            full_mask = causal_masked

        x = self.decoder(x, full_mask)
        logits = self.projection(x)

        return logits


def llm_model(vocab_size:int, seq_len:int, d_model:int, d_ff:int, h:int, N:int, dropout:float):
    embedding = Embedding(d_model, vocab_size)

    pos_en = PositionalEncoding(seq_len, d_model, dropout)

    decoder_block = []
    for _ in range(N):
        decoder_multiAttention = MultiHeadAttention(seq_len, d_model, dropout, h)
        decoder_feedForward = FeedForward(dropout, d_ff, d_model)
        decoderBlock = DecoderBlock(decoder_multiAttention, decoder_feedForward, d_model, dropout)
        decoder_block.append(decoderBlock)

    decoder = Decoder(nn.ModuleList(decoder_block), d_model)

    projectionLayer = ProjectionLayer(d_model, vocab_size)

    transformers = Transformers(embedding, pos_en, decoder, projectionLayer)

    for p in transformers.parameters():
        if p.dim() > 1:
            # Biases are usually 1D (like [512]) and are left alone - often initialised to zero by default
            # Weights have dimensions more than 1
            nn.init.xavier_uniform_(p)
            # Applies Xavier uniform initialisation to the tensor p
            # This method carefully sets the initial values of the weights to keep variance stable through the network

            # To prevent vanishing, exploding gradients and slow convergence or total training failure

    return transformers



