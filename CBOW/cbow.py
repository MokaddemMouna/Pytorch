
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np




CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 10

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
  
# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)
  
word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
      
 
  
class CBOW(nn.Module):
  
    def __init__(self,vocab_size,embdedding_dim):
        super(CBOW, self).__init__()
        self.emb = nn.Embedding(vocab_size,embdedding_dim)
        self.linear = nn.Linear(embdedding_dim,vocab_size)
          
  
    def forward(self, inputs):
          
#         print(inputs.shape)
        emb = self.emb(inputs)
#         print(emb.shape)
        emb = emb.mean(dim=0)
#         print(emb.shape)
        emb = emb.view((1,-1))
#         print(emb.shape)
        l = self.linear(emb)
#         print(l.shape)
        log_probs = F.log_softmax(l, dim=1)
#         print(log_probs.shape)
#         print('---------------')
        return log_probs
         
          
  
# create your model and train.  here are some functions to help you make
# the data ready for use by your module
  
  
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)
  
  
  
losses = []
loss_function = nn.NLLLoss()
model = CBOW(vocab_size,EMBEDDING_DIM)
optimizer = optim.SGD(model.parameters(),lr=0.001)
 
# train the model 
for epoch in range(100):
    total_loss = 0
    for context,target in data:
        idx = make_context_vector(context, word_to_ix)
          
          
        model.zero_grad()
          
        log_probs = model(idx)
          
        loss = loss_function(log_probs,torch.tensor([word_to_ix[target]], dtype=torch.long))
          
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
          
    losses.append(total_loss)
  
print(losses)
 
# test the model
with torch.no_grad():
    sample = data[3]
    print(sample)
    context = sample[0]
    idx = make_context_vector(context, word_to_ix)
    log_probs = model(idx)
    i = torch.argmax(log_probs, dim=1)
    print(i)
    print(word_to_ix)
