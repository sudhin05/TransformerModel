import torch
import torch.nn as nn
from torchinfo import summary
from math import sqrt


class SelfAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    """
      d_model actually represents the embedding size, 
      Note that the model accepts the input as a batch, sequence, d_model input
      n_heads represents the number of heads
      d_v represents the size of each head vector. Which would come down to sequence d_v
      d_v is also referred to as d_k in the paper
    """
    super(SelfAttention,self).__init__()
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_v = d_model // n_heads

    #256 embedding size split into 7 parts not possible
    # so we can assert head dimension * n_heads = embed size and if it doesnt we can say embed size needs to be divisible by heads

    assert d_model % n_heads == 0, "Embed size needs to be divisible by the n_heads"

    #Now initializing all the weight in mha self attention, ek toh q,k,v hai and we also have wo after the concatenation operation
    #I think it should be d_model instead of d_v here, but we'll see if that's the case or why not
    #So below would give of Q' , K' , V' and MHA
    self.queries = nn.Linear(self.d_v,self.d_v,bias=False)
    self.keys = nn.Linear(self.d_v,self.d_v,bias = False)
    self.values = nn.Linear(self.d_v,self.d_v,bias = False)
    self.fc = nn.Linear(self.d_v*n_heads,d_model)

  def forward(self,queries,keys,values,mask):
    #No. of training examoles(batch size)
    N = queries.shape[0]
    #Setting the length of queries, key, values which depends on the usage in encode or decoder, so they have to be set always
    #So technically these are sequence lengths
    query_len, key_len, value_len = queries.shape[1],keys.shape[1],values.shape[1]

    #Now Q' , K' , V' needs to be split into its heads
    #Earlier dimension was N,query_len,d_model
    queries = queries.reshape(N,query_len,self.n_heads,self.d_v )
    keys = keys.reshape(N,key_len,self.n_heads,self.d_v)
    values = values.reshape(N,value_len,self.n_heads,self.d_v)

    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)
    
    #Matrix multiplication methods
    #Using torch.einsum for our matrix multiplication 
    energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])

    #Or using batch matrix multiplication
    # n,h,q,d and n,h,d,k krdo Just check if both blocks work the same
    """
    Stack overflow discussion
      a = tf.random.normal(shape=[128,24,24,256])
      b = tf.random.normal(shape=[128,24,24,64])
      c = tf.random.normal(shape=[128,24,64,256])

      # Essentially performing a [24, 64] . [64, 256]
      # This is dot product
      bc = tf.matmul(b,c)

      # Res would be of size `128,24,24,256`
      # This is element-wise
      res = a * bc
    """

    # q = queries.permute(0,2,1,3)
    # k = keys.permute(0,2,3,1)
    # energy = torch.matmul(q,k)
    #energy is n,h,q,k now

    if mask is not None:
      """
        Tensor.masked_fill(mask, value)
        Fills elements of self tensor with value where mask is True. The shape of mask must be broadcastable with the shape of the underlying tensor.
        I have tensor named "k1" which is in shape 3,1,1,9 and also have p1 tensor in shape of 3,7,9,9 and I wanna know what does the line below do?

        p1 = p1 .masked_fill(k1== 0, float("-1e30"))
        In your case it will place in p1 the value of float("-1e30") at the positions where k1 is equal to zero. Since k1 has singleton dimensions its shape will be broadcasted to the shape of p1.

      """
      """
        mask == 0
        mask is typically a tensor of the same shape as energy, containing values of either 0 or 1.
        Positions in the mask with a value of 0 correspond to elements we want to ignore or "mask out" in the energy tensor.
        By evaluating mask == 0, we get a boolean tensor where True marks positions to be ignored and False marks positions to keep.
        masked_fill_ is an in-place operation (modifies energy directly) that replaces elements in energy where condition is True with the specified value.
        In this case, it sets positions where mask == 0 to -float("inf")    
      """
      # if the element of the mask is zero , we want to shut it down and replace it so it doesn't impact others
      energy = energy.masked_fill_(mask == 0, -float("inf"))
      #N,h,q,k hai energy ka shape toh we want to apply across dim = k
      """
        By applying softmax across k (the last dimension), we are normalizing the scores for each query position so that the weights across all key positions sum to 1.
        This aligns with the idea of attention: each query position focuses on different key positions, and softmax helps distribute this focus proportionally.
      """
      #Toh sort of query ko pta chl jaata hai over time uske liye kya zyada relatable ya important hai
      attention = torch.softmax(energy/(sqrt(self.d_model)),dim = -1)

      #we want out to be N,q,h,d
      output = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N,query_len,self.n_heads*self.d_v)
      # Now we have
      #alternate using cat operation too complicated

      output = self.fc(output)
      return output
    
class TransformerBlock(nn.Module):
  def __init__(self,d_model,n_heads,dropout,exp_factor):
    super(TransformerBlock,self).__init__()
    self.attention = SelfAttention(d_model,n_heads)
    """
      Applies Layer Normalization over a mini-batch of inputs.
      y = ((x-E[x])/sqrt(Var[x] + e)) * γ + β
      This layer implements the operation as described in the paper Layer Normalization
      The mean and standard-deviation are calculated over the last D dimensions, where D is the dimension of normalized_shape. For example, if normalized_shape is (3, 5) (a 2-dimensional shape), the mean and standard-deviation are computed over the last 2 dimensions of the input (i.e. input.mean((-2, -1))). 
      γ and β are learnable affine transform parameters of normalized_shape if elementwise_affine is True. The standard-deviation is calculated via the biased estimator, equivalent to torch.var(input, unbiased=False).
      
      # NLP Example
      batch, sentence_length, embedding_dim = 20, 5, 10
      embedding = torch.randn(batch, sentence_length, embedding_dim)
      layer_norm = nn.LayerNorm(embedding_dim)
      # Activate module
      layer_norm(embedding)
    """
    self.norm1 = nn.LayerNorm(d_model)
    """
    
      Pytorch containers: Sequential
      Useful for homogeneous layer organisation
      nn.Sequential is best suited for linear, straightforward layer stacks where layers are applied sequentially, 
      and the output of one layer directly becomes the input of the next. It simplifies the implementation by automatically chaining operations without needing a custom forward() method. 
      Use nn.Sequential when the model's flow is simple and fixed, such as in feed-forward networks or feature extraction pipelines in convolutional neural networks (CNNs).
    
      nn.Sequential is a container that automatically connects layers one after the other. When you pass an input,
      it runs through all the layers sequentially. Internally, the forward() method of nn.Sequential iterates through all its layers and 
      applies them in order

      When you call model(input), it automatically executes:
      input -> Linear(10, 20) -> ReLU() -> Linear(20, 30).

      Why Use It:
      If your model structure is fixed and linear (e.g., a feed-forward network), 
      nn.Sequential keeps your code clean and avoids writing a custom forward() method.

      Limitations:

      You cant add conditional logic (e.g., if conditions).
      It doesnt allow sharing layers or applying operations like concatenation or addition between layers.
    """
    self.feed_forward = nn.Sequential(
      nn.Linear(d_model,d_model*exp_factor),
      nn.ReLU(),
      nn.Linear(d_model*exp_factor,d_model)
    )
    self.norm2 = nn.LayerNorm(d_model)
    """
    
      During training, randomly zeroes some of the elements of the input tensor with probability p.

      The zeroed elements are chosen independently for each forward call and are sampled from a Bernoulli distribution.
      Each channel will be zeroed out independently on every forward call.

      This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper Improving neural networks by preventing co-adaptation of feature detectors .

      Furthermore, the outputs are scaled by a factor of 1/(1-p)
      during training. This means that during evaluation the module simply computes an identity function.
    
    """
    self.dropout = nn.Dropout(dropout)

  def forward(self,queries,keys,values,mask):
    attention = self.attention(queries,keys,values,mask)
    x1 = self.dropout(self.norm1(attention+queries))
    fc = self.feed_forward(x1)
    x2 = self.dropout(self.norm2(fc + x1))
    return x2

class Encoder(nn.Module):
  #Max length required to help with the padding of the input data
  def __init__(self,vocab_size,d_model,n_layers,n_heads,device,exp_factor,dropout,max_length):
    super(Encoder, self).__init__()
    self.d_model = d_model
    self.device = device
    self.n_layers = n_layers
    """
      A simple lookup table that stores embeddings of a fixed dictionary and size.

      This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
      # an Embedding module containing 10 tensors of size 3
      embedding = nn.Embedding(10, 3)
      # a batch of 2 samples of 4 indices each
      input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
      embedding(input)



      # example with padding_idx
      embedding = nn.Embedding(10, 3, padding_idx=0)
      input = torch.LongTensor([[0, 2, 0, 5]])
      embedding(input)

      # example of changing `pad` vector
      padding_idx = 0
      embedding = nn.Embedding(3, 3, padding_idx=padding_idx)
      embedding.weight
      with torch.no_grad():
          embedding.weight[padding_idx] = torch.ones(3)
      embedding.weight
    """
    self.embedding = nn.Embedding(vocab_size, d_model)  
    self.position_embedding = nn.Embedding(max_length,d_model)
    """
      An intersting discussion regarding sequential and moduleList:
      https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/3

      How it Works:
      nn.ModuleList stores layers in a Python list, but it does not define their execution order. 
      Unlike nn.Sequential, you need to write the forward() method yourself to specify how these layers are used.

      Why Use It:
      Use nn.ModuleList when:

      The model architecture involves loops or dynamic behavior.
      You want full control over how and when layers are applied (e.g., RNNs or transformers with repeated blocks).
      You need to append or modify layers dynamically during training or inference.
      Limitations:

      Layers dont execute automatically—you must explicitly call them in forward().
      It doesnT support layer names like nn.ModuleDict.

    """
    self.layers = nn.ModuleList([
      TransformerBlock(d_model,n_heads,dropout=dropout,exp_factor=exp_factor)
      for _ in range(n_layers)
    ])
    self.dropout = nn.Dropout(dropout)

  def forward(self,x,mask):
    N,seq_len = x.shape
    """
      Arrange
      Returns a 1-D tensor of size [(end-start)/step]
      with values from the interval [start, end) taken with common difference step beginning from start.
    
      Expand
      Tensor.expand(*sizes) → Tensor
      Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
      Passing -1 as the size for a dimension means not changing the size of that dimension.
      Tensor can be also expanded to a larger number of dimensions, and the new ones will be appended at the front. For the new dimensions, the size cannot be set to -1.
      Expanding a tensor does not allocate new memory, but only creates a new view on the existing tensor where a dimension of size one is expanded to a larger size by setting the stride to 0. Any dimension of size 1 can be expanded to an arbitrary value without allocating new memory.
    
      x = torch.tensor([[1], [2], [3]])
      x.size()
      x.expand(3, 4)
      x.expand(-1, 4)   # -1 means not changing the size of that dimension
    
    """
    positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
    out = self.dropout(self.embedding(x) + self.position_embedding(positions))

    for layer in self.layers:
      output = layer(out,out,out,mask)

    return output

class DecoderBlock(nn.Module):
  def __init__(self,d_model,n_heads,exp_factor,dropout,device):
    super(DecoderBlock, self).__init__()
    self.attention = SelfAttention(d_model,n_heads)
    self.norm = nn.LayerNorm(d_model)
    self.transformer_block = TransformerBlock(d_model,n_heads,dropout,exp_factor)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x,values,keys,src_mask,trg_mask):
    # trg_mask is necessary, src_mask is optional, we'll need to padding for sentences of diff lengths
    attention = self.attention(x,x,x,trg_mask)
    queries = self.dropout(self.norm(attention + x))
    out = self.transformer_block(queries,keys,values,src_mask)
    return out

# self,vocab_size,d_model,n_layers,n_heads,device,exp_factor,dropout,max_length
class Decoder(nn.Module):
  def __init__(self,trg_vocab_size,d_model,n_layers,n_heads,device,exp_factor,dropout,max_length ):
    super(Decoder,self).__init__()
    self.device = device
    self.embedding = nn.Embedding(trg_vocab_size,d_model)
    self.position_embedding = nn.Embedding(max_length,d_model)
    self.layers = nn.ModuleList([
      DecoderBlock(d_model,n_heads,exp_factor,dropout,device)
      for _ in range(n_layers)
    ])
    self.fc = nn.Linear(d_model,trg_vocab_size)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self,x,enc_out,src_mask,trg_mask):
    N,seq_len = x.shape
    positions = torch.arange(0,seq_len).expand(N,seq_len).to(self.device)
    x = self.dropout(self.embedding(x) + self.position_embedding(positions))

    for layer in self.layers:
      x = layer(x,enc_out,enc_out,src_mask,trg_mask)
    out = self.fc(x)
    return out
  
class Transformer(nn.Module):
  def __init__(self,src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,d_model=512,n_layers=6,n_heads=8,device='cpu',exp_factor=4,dropout=0.1,max_length=100):
    super(Transformer, self).__init__()
    self.device = device
    self.encoder = Encoder(src_vocab_size,d_model,n_layers,n_heads,device,exp_factor,dropout,max_length)
    self.decoder = Decoder(trg_vocab_size,d_model,n_layers,n_heads,device,exp_factor,dropout,max_length)
    self.src_pad_idx = src_pad_idx
    self.trg_pad_idx = trg_pad_idx

  def make_src_mask(self,src):
    # N,1,1,src_len
    src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask.to(self.device)
  
  def make_trg_mask(self,trg):
    N,trg_len = trg.shape
    
    """
      Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.

      The lower triangular part of the matrix is defined as the elements on and below the diagonal.

      The argument diagonal controls which diagonal to consider. If diagonal = 0, all elements on and below the main diagonal are retained. 
      A positive value includes just as many diagonals above the main diagonal, and similarly a negative value excludes just as many diagonals below the main diagonal. 
    """
    trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(N,1,trg_len,trg_len)

    return trg_mask.to(self.device)
  
  def forward(self,src, trg):
    src_mask = self.make_src_mask(src)
    trg_mask = self.make_trg_mask(trg)
    enc_out = self.encoder(src, src_mask)
    out = self.decoder(trg, enc_out, src_mask, trg_mask)
    return out
  
if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  x = torch.tensor([[1,5,6,2,4,8,9,3,0],[1,3,4,6,7,4,9,8,2]]).to(device)
  trg = torch.tensor([[1,2,3,4,5,6,7,0],[1,3,4,5,6,7,8,9]]).to(device)
  src_pad_idx = 0
  trg_pad_idx = 0 
  src_vocab_size = 10
  trg_vocab_size = 10
  model = Transformer(src_vocab_size, trg_vocab_size,src_pad_idx,trg_pad_idx).to(device)
  # out = model(x,trg[:,:-1])
  # print(trg[:,:-1].dtype)
  # print(x.dtype)
  # print(out.shape)
  # print(model)
  summary(model,[(2,9),(2,7)],dtypes = [torch.int64,torch.int64])


 



    








    



 