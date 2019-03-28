'''
Created on Mar 27, 2019

@author: mmokadde
'''

# ------------------ Augmenting the LSTM part-of-speech tagger with character-level features

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import Counter
import string

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_sequence_for_char(sent,to_ix):
    sent_list = []
    for word in sent:
        word_list = []
        for char in word:
            idx = to_ix[char]
            word_list.append(idx)
        sent_list.append(torch.tensor(word_list, dtype=torch.long))
    return sent_list
            
 
 
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = Counter()
for sent,tags in training_data:
    for word in sent:
        word_to_ix[word] = len(word_to_ix)
        
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

alphabet = string.ascii_lowercase + string.ascii_uppercase
char_to_ix = Counter()
for char in alphabet:
    char_to_ix[char] = len(char_to_ix) + 1

char_to_ix['PAD'] = 0    


CHAR_VOCAB_SIZE = len(char_to_ix); CHAR_EMBEDDING_DIM = 10; CHAR_HIDDEN_DIM = 7;
VOCAB_SIZE = len(word_to_ix); TAGSET_SIZE = len(tag_to_ix); WORD_EMBEDDING_DIM = 6; WORD_HIDDEN_DIM = 6;  



class LSTMTaggerWithCharEmb(nn.Module):
 
    def __init__(self, char_vocab_size,char_embedding_dim,char_hidden_dim,vocab_size,tagset_size,
                 word_embedding_dim,word_hidden_dim):
        super(LSTMTaggerWithCharEmb, self).__init__()
        
        self.char_hidden_dim = char_hidden_dim
        
        self.char_embedding = nn.Embedding(num_embeddings=char_vocab_size,embedding_dim=char_embedding_dim)
        
        self.char_lstm = nn.LSTM(input_size=char_embedding_dim,hidden_size=char_hidden_dim)
        
        self.word_embedding = nn.Embedding(num_embeddings= vocab_size,embedding_dim=word_embedding_dim)
        
        self.word_lstm = nn.LSTM(input_size=word_embedding_dim+char_hidden_dim,hidden_size=word_hidden_dim)
        
        self.hidden2tag = nn.Linear(word_hidden_dim, tagset_size)
        
 
    def forward(self, sentence_word,sentence_char):
        
#         print(sentence_char)
        char_embeddings = self.char_embedding(sentence_char)
#         print(char_embeddings.shape)
        char_embeddings = char_embeddings.view(len(sentence_char[0]),len(sentence_char),-1)
#         print(char_embeddings.shape)
        _,(char_encoding,c_len) = self.char_lstm(char_embeddings)
#         print(char_encoding.shape)
        char_encoding = char_encoding.view(len(sentence_char),self.char_hidden_dim)
#         print(char_encoding.shape)
        
        word_embeddings = self.word_embedding(sentence_word)
#         print(word_embeddings.shape)
        concat = torch.cat((word_embeddings,char_encoding),1)
#         print(concat.shape)
        lstm_out, _ = self.word_lstm(concat.view(len(sentence_word),1,-1))
#         print(lstm_out.shape)
        
        tag_space = self.hidden2tag(lstm_out.view(len(sentence_word), -1))
#         print(tag_space.shape)
        tag_scores = F.log_softmax(tag_space, dim=1)
#         print(tag_scores.shape)
        return tag_scores  

 
  
    
# train the model
model = LSTMTaggerWithCharEmb(CHAR_VOCAB_SIZE,CHAR_EMBEDDING_DIM,CHAR_HIDDEN_DIM,VOCAB_SIZE,TAGSET_SIZE,
                   WORD_EMBEDDING_DIM,WORD_HIDDEN_DIM)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
 
# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)                           
                                         
                                         
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        
        # Step 2. Get our inputs ready for the network, that is, turn them into 
        # Tensors of char indices. 
        # Then padd the list of tensors so every sequence of chars will have the same length
        
        char_sentence_in = prepare_sequence_for_char(sentence,char_to_ix)
        char_sentence_in_padded = torch.nn.utils.rnn.pad_sequence(char_sentence_in,batch_first=True)
        
 
        # Step 3. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        
 
        # Step 4. Run our forward pass.
        tag_scores = model(sentence_in,char_sentence_in_padded)
 
        # Step 5. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()    
        
        
# See what the scores are after training
with torch.no_grad():
    word_inputs = prepare_sequence(training_data[0][0], word_to_ix)
    char_inputs = prepare_sequence_for_char(training_data[0][0],char_to_ix)
    char_inputs_padded = torch.nn.utils.rnn.pad_sequence(char_inputs,batch_first=True)
    tag_scores = model(word_inputs,char_inputs_padded)
      
    # wanted to try an unseen vocab example
#     inputs = prepare_sequence('Mouna ate pineapple'.split(), word_to_ix)
#     tag_scores = model(inputs)
      
    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)                                     
                                         
                                         
                                         
        
