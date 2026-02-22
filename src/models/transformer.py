import torch
import torch.nn as nn
from src.models.attention import MultiHeadAttention
from src.models.embeddings import PositionalEncoding, TokenEmbedding

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout_p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_p):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_p)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)
        self.sublayer = nn.ModuleList([AddNorm(d_model, dropout_p) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)[0])
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout_p):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout_p)
        self.encoder = Encoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size) # For language modeling

    def forward(self, src, src_mask):
        src = self.token_embedding(src)
        src = self.positional_encoding(src)
        encoded_output = self.encoder(src, src_mask)
        output = self.output_layer(encoded_output)
        return output
