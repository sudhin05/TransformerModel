import torch
import torch.nn as nn
from transformer import Encoder

vocab_size = 10000
max_length = 50
batch_size = 4
d_model = 512
n_layers = 2
n_heads = 8
exp_factor = 4
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Returns a tensor filled with random integers generated uniformly between low (inclusive) and high (exclusive).
x = torch.randint(0, vocab_size, (batch_size, max_length)).to(device)
# print(x.shape)
mask = torch.ones(batch_size, n_heads, max_length, max_length).to(device)
mask[0, 0, 2, 5] = 0 

encode = Encoder(vocab_size,d_model,n_layers,n_heads,device,exp_factor,dropout,max_length).to(device)

output = encode(x,mask)
print(output.shape)