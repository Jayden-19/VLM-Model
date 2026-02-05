import torch
import torch.nn as nn
import math

class LinearProjection(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size

        self.conv = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            # Patch size = (16 x 16 = 256)
            # Current channel: 3
            # 3 x 256 = 768
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # Original image size: [Batch_size, 3, 1024, 1024]
        x = self.conv(x)
        # Current Shape: [Batch_size, 768, 64, 64]

        # Flatten dimension
        x = torch.flatten(x, 2)
        # [Batch_size, 768, 64*64]

        x = torch.transpose(1,2)
        # [Batch_size, 4096, 768]
        # Match the size needed for LLM [Batch_size, seq_len, d_model]
        # d_model is the size of the vector to describe a word

        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1,seq_len+1, d_model))
        self.extra_embedding = nn.Parameter(torch.randn(1, 1 ,d_model))

    def forward(self, x):
        batch_size = x.shape[0]
        extra_embedding = self.extra_embedding.expand(batch_size, -1, -1)
        # Expand to allow the extra embedding to have the same shape as x to enable concatination
        x = torch.concat([extra_embedding, x], 1)
        x += self.positional_encoding
        return x
    
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        mean = torch.mean(x,-1, keepdim=True)
        std = torch.std(x,-1, keepdim=True)

        x = ((x-mean))/(std + 1e-6)*self.gamma+self.beta
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h = h
        self.d_model = d_model

        assert self.d_model % self.h == 0 ,f"{self.d_model} is not divisible by {self.h}"
        self.d_k = self.d_model // self.h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention_mechanism(q,k,v):
        d_k = q.shape[-1]
        attention_scores = (q @ torch.transpose(k, -1,-2)) / math.sqrt(d_k)

        attention_scores = attention_scores.softmax(dim=-1)

        return attention_scores @ v
    
    def forward(self, x):
        query = x
        key = x
        value = x

        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        query = query.view(query.shape[0], query.shape[1], self.h, -1).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, -1).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, -1).transpose(1,2)

        attention = self.attention_mechanism(query, key, value)

        attention = attention.transpose(1,2).contiguous().view(x.shape[0], x.shape[1],-1)
        # Use contiguous to create a copy of the transposed attention needed for the view

        return self.w_o(attention)
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.sequential(x)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.multihead = MultiHeadAttention(d_model, h)
        self.mlp = FeedForwardNetwork(d_model, 0.1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        old_x = x
        x = self.norm1(x)
        x = self.multihead(x)
        x = self.dropout(x)

        x = x + old_x

        old_x = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + old_x 
        return x
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, llm_d_model):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, llm_d_model),
            nn.GELU(),
            nn.Linear(llm_d_model, llm_d_model)
        )

    def forward(self, x):
        return self.projector(x)
    
class FullViT(nn.Module):
    def __init__(self, linearProjection: LinearProjection, encoder: nn.ModuleList, bridge: ProjectionLayer):
        super().__init__()
        self.linear_projection = linearProjection
        self.encoder = encoder
        self.projection_bridge = bridge
    
    def forward(self, x):
        x = self.linear_projection(x)
        for layer in self.encoder:
            x = layer(x)
        return self.projection_bridge(x)
    
def get_vit(img_size: int, patch_size: int, in_chans: int, d_model: int, h: int, dropout: float, llm_d_model: int, N: int):
    linear_projection = LinearProjection(img_size, patch_size, in_chans, d_model)

    encoder_block = []
    for _ in range(N):
        encoder_block.append(TransformerEncoder(d_model, h, dropout))

    encoder_block = nn.ModuleList(encoder_block)

    projection_bridge = ProjectionLayer(d_model, llm_d_model)

    vit_block = FullViT(linear_projection, encoder_block, projection_bridge)

    for p in vit_block.parameters():
        if p.dim() > 1:
            # Biases are usually 1D (like [512]) and are left alone - often initialised to zero by default
            # Weights have dimensions more than 1
            nn.init.xavier_uniform_(p)
            # Applies Xavier uniform initialisation to the tensor p
            # This method carefully sets the initial values of the weights to keep variance stable through the network

            # To prevent vanishing, exploding gradients and slow convergence or total training failure

    return vit_block