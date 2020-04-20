This is the implementation of **Bahdanau**'s attention [paper](https://arxiv.org/abs/1409.0473).

I didn't implement the code from scratch, I followed a tutorial [here](https://colab.research.google.com/github/bastings/annotated_encoder_decoder/blob/master/annotated_encoder_decoder.ipynb#scrollTo=gbnrkdh0w4XW). I read the code, analyzed it and added several comments. 

The first attempt was to follow Pytorch's official [tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html?fbclid=IwAR1hT0w0Yfnf5aZ27S6J7rH_vRkjbu7XgD7Yw43z9EGwLrsAawjPaYAe-sk) on attention but the implementation is different from the original paper. Among many differences, the major one was not to include the encoder's hidden layers in the calculation of the attention
weights.
Some other links that may be helpful to understand more the concepts are in the following:
- Jal Ammar's [blog](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/?fbclid=IwAR0qRBq-LfcgwkZrw7FgsSjXDOzHw3IURBaDOcPZVOLIvDoaNbpvjcIcRNs)
- Khandelwel's [blog](https://towardsdatascience.com/sequence-2-sequence-model-with-attention-mechanism-9e9ca2a613a) on differences between **Bahdanau** and **Luong** attention mecanisms.


In the code there is use of masks for both target and source sentences. Those masks mark the positions of the padding indices.
They are stored in two variables and used for different goals:
- **trg_mask**: it expresses the real length of the target sentence without the padding which is used to determine the length for unrolling the decoder states si. Note that obviously, it is used only in training. In test, we take another variable called **max_len**, for example in this code it is value is 100. This will affect prediction time. In fact, we need to choose a value sufficiently big so it fits the longest translated sentence in test time, at the same time sentences in general tend to be shorter than that and we end up doing many useless iterations (because they will alternate between predicting '.' and eos). 
- **src_mask**: vector that identifies padding positions (same as trg_mask) in order to assign to them a value of -inf in the calculated energy scores in Bahdanau attention. In other words, alpha_i(s) of padding positions will be -inf. 
