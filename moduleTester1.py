import torch
import torch.nn as nn
from transformer import SelfAttention

d_model = 256  
n_heads = 8   
batch_size = 4 
seq_len = 10  

queries = torch.rand(batch_size, seq_len, d_model)
keys = torch.rand(batch_size, seq_len, d_model)
values = torch.rand(batch_size, seq_len, d_model)

mask = torch.ones(batch_size, n_heads, seq_len, seq_len)
mask[0, 0, 2, 5] = 0  

model = SelfAttention(d_model = d_model, n_heads = n_heads)
output = model(queries, keys, values, mask)

print(output.shape)
