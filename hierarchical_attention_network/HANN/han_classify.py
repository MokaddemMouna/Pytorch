import torch
from torch import nn
from HANN.utils import preprocess, rev_label_map
from tqdm import tqdm
import json
import os
import pandas as pd
from collections import OrderedDict
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best model
checkpoint = 'checkpoint_han.pth.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval() #test mode

# Pad limits, can use any high-enough value since our model does not compute over the pads
sentence_limit = 15
word_limit = 20

# Word map to encode with
data_folder = './han_data'
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()

def read_client_csv(csv_file):

    """
    Read the loaded csv file of the client

    :param document: a csv file
    :return: list of documents, each document is a list of tokenized sentences

    """

    docs = []
    labels = []
    data = pd.read_csv(csv_file, header=None)
    for i in range(data.shape[0]):
        row = list(data.loc[i, :])

        sentences = list()

        for text in row[1:]:
            for paragraph in preprocess(text).splitlines():
                sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

        words = list()
        for s in sentences[:sentence_limit]:
            w = word_tokenizer.tokenize(s)[:word_limit]
            # If sentence is empty (due to removing punctuation, digits, etc.)
            if len(w) == 0:
                continue
            words.append(w)
        # If all sentences were empty
        if len(words) == 0:
            continue

        docs.append(words)

    return docs

def classify(document):
    """
    Classify a document with the Hierarchial Attention Network (HAN).

    :param document: a document in text form
    :return: the predicted topic
    """

    # Number of sentences in the document
    sentences_in_doc = len(document)
    sentences_in_doc = torch.LongTensor([sentences_in_doc]).to(device)  # (1)

    # Number of words in each sentence
    words_in_each_sentence = list(map(lambda s: len(s), document))
    words_in_each_sentence = torch.LongTensor(words_in_each_sentence).unsqueeze(0).to(device)  # (1, n_sentences)

    # Encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            document)) + [[0] * word_limit] * (sentence_limit - len(document))
    encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

    # Apply the HAN model
    scores, word_alphas, sentence_alphas = model(encoded_doc, sentences_in_doc,
                                                 words_in_each_sentence)  # (1, n_classes), (1, n_sentences, max_sent_len_in_document), (1, n_sentences)
    scores = scores.squeeze(0)  # (n_classes)
    scores = nn.functional.softmax(scores, dim=0)  # (n_classes)

    # Find best prediction
    score, prediction = scores.max(dim=0)
    prediction = rev_label_map[prediction.item()]

    return prediction, round(score.item() * 100,2)

def classify_documents(file):

    """
        Classify a file of documents with the Hierarchial Attention Network (HAN).

        :param documents: a file containing several documents in the same form as the yahoo_answers dataset
        :return: the predicted topics
        """

    documents = read_client_csv(file)

    # a dict to store the predictions of the documents
    predictions = OrderedDict()
    for i, d in enumerate(tqdm(documents,desc='Predicting')):
        print("\n",d)
        predictions[f"document{i}"] = list(classify(d))
        print(predictions[f"document{i}"])
    return predictions


