This is the implementation of **Bahdanau**'s attention [paper](https://arxiv.org/abs/1409.0473).

I didn't implement the code from scratch, I followed a tutorial [here](https://colab.research.google.com/github/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb#scrollTo=gbnrkdh0w4XW). I read the code, analyzed it and added several comments. 

The first attempt was to follow Pytorch's official [tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html?fbclid=IwAR1hT0w0Yfnf5aZ27S6J7rH_vRkjbu7XgD7Yw43z9EGwLrsAawjPaYAe-sk) on attention but the implementation is different from the original paper. Among many differences, the major one was not to include the encoder's hidden layers in the calculation of the attention
weights.
Some other links that may be helpful to understand more the concepts are in the following:
- Jal Ammar's [blog](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/?fbclid=IwAR0qRBq-LfcgwkZrw7FgsSjXDOzHw3IURBaDOcPZVOLIvDoaNbpvjcIcRNs)
- Khandelwel's [blog](https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a) on differences between **Bahdanau** and **Luong** attention mecanisms.
