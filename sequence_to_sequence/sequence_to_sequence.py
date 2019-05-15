'''
Created on Apr 12, 2019

@author: mmokadde
'''
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1



class Lang:
    def __init__(self,name):
        
        self.name = name
        self.word_to_index = {}
        self.index_to_word = {0:'SOS_token',1:'EOS_token'}
        self.word_count = {}
        self.n_words = 2
        
    def add_word(self,word):
        
        if word not in self.word_to_index:
            self.word_to_index[word] = self.n_words
            self.index_to_word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
            
    def add_sentence(self,sentence):
        
        for word in sentence.split(' '):
            self.add_word(word)
            

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1,lang2,reverse=False):
    
    pairs = []
    file = 'data/{}-{}.txt'.format(lang1,lang2)
    lines = open(file, encoding='utf-8').read().strip().split('\n')
    
    pairs = [[ normalize_string(s) for s in l.split('\t')] for l in lines]
    
    if reverse:
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
        pairs = [list(reversed(p)) for p in pairs]
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
        
    return input_lang,output_lang,pairs
        
    
    
    print('m')
        
# read_langs('eng', 'fra')            
        

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(pair):
     
    input_pair = pair[1]
    return len(pair[0].split(' ')) <= MAX_LENGTH and \
    len(input_pair.split(' ')) <= MAX_LENGTH and \
    input_pair.startswith(eng_prefixes)

def filter_pairs(pairs):
    return [p for p in pairs if filter_pair(p)]


def prepare_data(lang1, lang2, reverse=False):
    
    input_lang,output_lang,pairs = read_langs(lang1, lang2, reverse)
    print("Read {} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data('eng', 'fra', True)

def indices_from_sentence(lang,sentence):
        return [lang.word_to_index[word] for word in sentence.split(' ')]
    
def tensor_from_sentence(lang,sentence):
    indices = indices_from_sentence(lang,sentence)
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=device)

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang,pair[0])
    target_tensor = tensor_from_sentence(output_lang,pair[1]).view(-1,1)
    return (input_tensor,target_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, dict_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(dict_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        
    def forward(self,input):
        
#         print(input.shape)
        embeds = self.embedding(input)
#         print(embeds.shape)
        output = embeds.view(len(input),1,-1)
#         print(output.shape)
        output,hidden = self.gru(output)
#         print(output.shape,hidden.shape)
        
        return output,hidden
    
    
    
class DecoderRNN(nn.Module):
    def __init__(self,dict_size,hidden_size):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(dict_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size)
        self.linear = nn.Linear(hidden_size,dict_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,input,hidden):
#         print(input.shape)
        embeds = self.embedding(input).view(1,1,-1)
#         print(embeds.shape)
        output = F.relu(embeds)
#         print(output.shape)
        output,hidden = self.gru(output,hidden)
#         print(output.shape)
        output = self.linear(output)
#         print(output.shape)
        output = self.softmax(output[0])
#         print(output.shape)
        return output,hidden
        
        
        
        
        
teacher_forcing_ratio = 0.5        
    
    

def train(n_epochs,n_iters,pairs,hidden_size,learning_rate):
    
    
    encoder = EncoderRNN(input_lang.n_words,hidden_size)
    decoder = DecoderRNN(output_lang.n_words,hidden_size)
    
    loss_function = nn.NLLLoss()
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    accumulated_loss = 0
    losses = []
    
    
    
    for i in range(n_epochs):
        accumulated_loss = 0
        training_pairs = [tensors_from_pair(random.choice(pairs)) for _ in range(n_iters)]
        for pair in training_pairs:
            
            loss = 0
            
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            input_tensor = pair[0]
            target_tensor = pair[1]
            target_length = target_tensor.shape[0]
            
            encoder_output,encoder_hidden = encoder(input_tensor)
            
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden
            
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            
            if use_teacher_forcing:
                for di in range(target_length):
                    real_target = target_tensor[di]
                    decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
                    loss += loss_function(decoder_output,real_target)
                    decoder_input = real_target
                
            else:
                for di in range(target_length):
                    decoder_output,decoder_hidden = decoder(decoder_input,decoder_hidden)
                    loss += loss_function(decoder_output,target_tensor[di])
                    top_value,top_index = decoder_output.topk(1)
                    decoder_input = top_index.squeeze().detach()
                    if decoder_input.item() == EOS_token:
                        break
                    
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            accumulated_loss += loss.item()/target_length
        
        avg_loss_over_examples = accumulated_loss/len(training_pairs)
        losses.append(avg_loss_over_examples)
        print('epoch {}: {}'.format(i,avg_loss_over_examples))
            
    

                
                
            

            
n_epochs = 3
n_iters =  75000
hidden_size = 256
learning_rate = 0.01

train(n_epochs,n_iters,pairs,hidden_size,learning_rate)
    
    
    
    
        
    
    
    
    
         
            
        
        
        
                
