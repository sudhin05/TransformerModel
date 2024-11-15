import torch
import torch.nn as nn
from transformer import TransformerBlock

d_model = 512
n_heads = 8
dropout = 0.1
exp_factor = 4
batch_size = 4 
seq_len = 32

queries = torch.rand(batch_size,seq_len,d_model)
keys = torch.rand(batch_size,seq_len,d_model)
values = torch.rand(batch_size,seq_len,d_model)

mask = torch.ones(batch_size, n_heads, seq_len, seq_len)
mask[0, 0, 2, 5] = 0 

model = TransformerBlock(d_model,n_heads,dropout,exp_factor)

output = model(queries,keys,values,mask)
print(output.shape)
