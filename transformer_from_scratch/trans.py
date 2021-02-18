import torch
import torch.nn as nn
import numpy as np
from torch.types import Device

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        '''
        param embed_size: embedding size
        param heads: split into 8 parts with 32 each (if embed_size 256 -> 256/8 = 32 parts)
        '''
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads
        
        assert (heads * self.head_dim == embed_size), "Int division error"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias =False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias =False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
         N = query.shape[0]
         value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
         #Now split into self.heads

         values = values.reshape(N, value_len , self.heads, self.head_dim)
         keys = keys.reshape(N, key_len, self.heads, self.head_dim)
         queries = query.reshape(N, query_len, self.heads, self.head_dim)
         
         values = self.values(values)
         keys = self.keys(keys)
         queries = self.queries(queries)
         energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys]) #(N, heads, query_len, key_len)
         if mask is not None:
             energy = energy.masked_fill(mask==0, float("1e-20")) # fix
         attention = torch.softmax(energy/ (np.sqrt(self.embed_size)), dim = 3)
         out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N,query_len, self.heads*self.head_dim)
         #att shape (N, heads, query_len, key_len)
         #val shape (N, val_len, heads, heads_dim)
         #out (N, query_len, heads, head_dim)
         out = self.fc_out(out)
         return out 


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, drop, forward_exp):
        super().__init__()
        self.att = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_exp*embed_size), nn.ReLU(), nn.Linear(forward_exp*embed_size, embed_size))
        self.drop = nn.Dropout(drop)

    def forward(self, value, key, query, mask):
        att = self.att(value, key, query, mask)
        x = self.drop(self.norm1(att + query))
        forward = self.feed_forward(x)
        out = self.drop(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_exp, drop, max_len):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embed = nn.Embedding(src_vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_len, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, drop, forward_exp)
        ])
        self.drop = nn.Dropout(drop)

    def forward(self, x, mask):
        N,seq_len = x.shape
        posit = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        out = self.drop(self.word_embed(x) + self.pos_emb(posit))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward, drop,device):
        super().__init__()
        self.att = MultiHeadAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.trans_block = TransformerBlock(embed_size, heads, drop, forward)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x,value, key, src_mask, trg_mask):
        att = self.att(x, x,x,trg_mask)
        query = self.drop(self.norm(att + x))
        out = self.trans_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward, drop, device, max_len):
        super().__init__()
        self.device = device
        self.word_emb = nn.Embedding(trg_vocab_size, embed_size)
        self.pos_emb = nn.Embedding(max_len, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, heads, forward, drop, device) for i in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.drop((self.word_emb(x) + self.pos_emb(pos)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size = 256, num_layers = 6, forward_exp=4, heads=8, drop = 0, device = "cuda", max_len = 100):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers,heads, device, forward_exp, drop,max_len)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers,heads, forward_exp, drop, device, max_len)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unqueeze(1).unsqueeze(2) #(N,1,1,src_len)
        return src_mask.to(self.device)
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len) #triangular matrix
        return trg_mask.to(self.device)

    def forward(self, src,trg):
        src_mask = self.make_src_mask
        trg_mask = self.make_trg_mask
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

device = torch.device("cuda")
x = torch.tensor([[1,5,6,4,3,9,5,2,0], [1,8,7,3,4,5,6,7,2]]).to(device)
trg = torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device)
src_pad_idx = 0
trg_pad_idx = 0
src_vocab_size = 10
trg_vocab_size = 10
model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
out = model(x, trg[:, :-1])
print(out.shape)