
This is an answer of the exercice given by Robert Guthrie in his [Deep Learning For NLP With Pytorch](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html#exercise-computing-word-embeddings-continuous-bag-of-words).


[Here](https://cs224d.stanford.edu/lecture_notes/notes1.pdf ) is a good explanation of theoretical aspects of word2vec. 

According to the lecture notes, cbow model will learn embeddings by trying to predict the word from its context.

Steps: 

1. Generate one hot word vectors of size the vocab size for the input context of size m (in our case it is 4 context words).

2. Get embedded word vectors for the context by multiplying the embedding matrix (first matrix of parameters) by everyone hot word vector.
In our case, we should get 4 vectors, each vector is the result of the previous multiplication. 

3. Average these vectors to get a vector of size the embedding size. 

4. Multiply vector of 3. with second matrix of parameters and then apply softmax to generate the output, which is a vector of probabilities, each one giving the probability of every word of the vocabulary to be the right prediction of the input context.


### Issue ?

Linear transformation [nn.linear()](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear) of Pytorch takes a vector as an input
but in our case the multiplication of 2. gives a matrix. 

### Solution:

Hugo Larochelle in his [video lecture](https://www.youtube.com/watch?v=PKszi8iogak) has explained how we can backpropagate through
the not only parameters but the input of the model given that the input of the new model is C(w) where C is the embedding matrix.

So using the [embedding layer](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) of Pytorch, we will get a tensor of dimension
[context size * 2,embedding size], then we will take the mean along axis=1 (columns), this will result in a tensor of size [embedding size]
, we have to transform that to tensor of size [1,embedding size] (with tensor.view()) in order to pass it to a linear transformation
with nn.linear(). And we need to backpropagate over this C, as explained in the video.
