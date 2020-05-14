
Reminder of the task:

- Enrich the word embeddings of the previously implemented Part Of Speech Tagger with Character embedding. 
- The character embedding is created using another LSTM.


## Version 1

##### Issues:

There are two main points I spent time thinking about to solve this task. Here are they:

1. Are we going to train the character embedding LSTM in a unsupervised fashion so to get the character embedding representation?
Something similar to the CBOW model? So, we will take context characters and try to predict the other characters. I didn't find
that this is a good idea because the length of words is not always the same, so if we fix the number of characters we try to learn
will they be the middle ones, is 1 or 2 or 3 characters are enough? And so many questions. 
<br/><br/>The second idea is to learn character embedding while we perform the pos tagging task like for the word embeddings. So here the 
character embedding lSTM will input a word (a character for each time step) and then the last hidden layer will be the character
level representation which we will concatenate to the word embedding of the corresponding word. <br/><br/><br/>


2. The second problem was of inputs and dimensions. In the first model, we process sentence by sentence. For an input sentence,
the word embedding of each word is calculated by the embedding layer then passed to the LSTM so here the LSTM input sequence is 
the sequence of words and the input dimension is (1,sequence length,input dimension) where the input dimension is 
the embedding dimension and 1 is the batch.
<br/><br/>Character level representation is different because each sentence is a sequence 
of words and each word is a sequence of characters. So if we want to process the characters for each sentence sample, 
the input will be of 4 dimensions (1,word sequence length,character sequence length,input dimension) where the input dimension here
is the character level embedding.
<br/><br/>So, in order to calculate character embedding, we need an embedding layer that outputs a 4-dimension output and a LSTM layer 
that accepts a 4-dimension input. Looking at Pytorch doc for [embedding layer](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding), you will find that 
embedding layer output is (*, embedding_dim), where * is the input shape. But for [LSTM layer](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM)
and any RNN layer generally in Pytorch, it accepts only input of 3 dimensions (seq_len, batch, input_size). And for that case,
[Here](https://discuss.pytorch.org/t/4-dimension-input-to-lstm/27119) is a suggested solution.
<br/><br/>Basically here, we will choose option 1 of the suggested solution. Why? Because in terms of character level representations the sentences are independent in the
sense that taking a word from sentence 1 or sentence 2 does not make any difference except for the word itself if it is different. 
And so the input of the character level LSTM will be now (1 * word sequence length,character sequence length,input dimension) and
so for each sentence, the character level LSTM will process a batch of words and will output the same batch of hidden layer, each
is the character level representation of each word of the sentence and that we will concatenate with the corresponding word 
embedding. <br/><br/> 

Something to note here is since the character level LSTM processes a batch of words, all word examples need to have the same length.
So here we choose a fixed length, which is the length of the longest word, and apply padding for all other words.

##### Useful links:

[Enhancing LSTMs With Character Embeddings For Named Entity Recognition (Keras)](https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/)

[Pytorch 4 dimension input to LSTM](https://discuss.pytorch.org/t/4-dimension-input-to-lstm/27119)<br/><br/>



## Version 2

In Version 1 of the model, we are applying padding to the input of character level representation LSTM. So implicitly the LSTM
is learning the representation of the '<PAD>', the padding character, which biases the model. So, what we need here is 
formula on how to apply a LSTM with variable size inputs. Luckily, Pytorch solves this problem with 
[pack_padded_sequence method](https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence). If you want to 
learn how to use it, I provide below a very good tutorial on packing and unpacking sequences in Pytorch. 


##### Useful link:

[Minimal tutorial on packing and unpacking sequences in pytorch](https://github.com/HarshTrivedi/packing-unpacking-pytorch-minimal-tutorial)
