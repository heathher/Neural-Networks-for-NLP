from random import random
import string
import re
from collections import defaultdict
import math

VOCAB_SIZE = 2500
ALPHA = 0.1

def count_labels(labels):
    return {label: sum(1 for l in labels if l == label) for label in set(labels)}

def tokenize(text):
    text = text.lower();
    replace_punctuation = str.maketrans("\"!#$%&'()*+,-./:;<=>?@[\]^_`{|}~¡¢£¤¦§¨«­®°³´·º»½¾¿–‘’“”…₤", 
        ' '*len("\"!#$%&'()*+,-./:;<=>?@[\]^_`{|}~¡¢£¤¦§¨«­®°³´·º»½¾¿–‘’“”…₤"));
    text = text.translate(replace_punctuation)
    return re.split("\W+", text)

def make_vocab(tokenized_texts):
    with open("Starter code/stopwords.txt", "r") as f:
        STOPWORDS = map(lambda x: x.strip(), f.readlines())
    STOPWORDS = set(STOPWORDS)
    word_dict = {}
    vocab = []
    for tokenized_text in tokenized_texts:
        for token in tokenized_text:
            if (token not in STOPWORDS):
                word_dict[token] = word_dict.get(token, 0) + 1
    sorted_vocab = sorted(word_dict.items(), key=lambda x:x[1], reverse = True)
    for idx in range(VOCAB_SIZE):
        vocab.append(sorted_vocab[idx][0])
    vocab = set(vocab)
    return vocab

def update_class_word_counter(tokens, class_word_counter, vocab):
    word_counter = 0
    for token in tokens:
        if token in vocab:
            class_word_counter[token] = class_word_counter.get(token, 0) + 1
            word_counter += 1
    return word_counter

def train(train_texts, train_labels):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """
    positive_class_word_counter = defaultdict(int)
    negative_class_word_counter = defaultdict(int)
    pos_word_counter = 0
    neg_word_counter = 0
    tokenized_train_text = [tokenize(r) for r in train_texts]
    my_vocab = make_vocab(tokenized_train_text)
    for text, label in zip(tokenized_train_text, train_labels):
        if label == 'neg':
            neg_word_counter += update_class_word_counter(text, negative_class_word_counter, my_vocab)
        else:
            pos_word_counter += update_class_word_counter(text, positive_class_word_counter, my_vocab)
    params = [pos_word_counter, neg_word_counter, positive_class_word_counter, negative_class_word_counter]
    return params
   

def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
    tokens = []
    for r in texts:
        tokens.append(tokenize(r))
    pos_count = params[0]
    neg_count = params[1]
    positive_class_word_counter = params[2]
    negative_class_word_counter = params[3]
    probalities = []    
    for text in tokens:
        p_pos = math.log(0.5)
        p_neg = math.log(0.5)
        for word in text:
            word_p_pos = positive_class_word_counter.get(word, 1.0) + ALPHA
            word_p_neg = negative_class_word_counter.get(word, 1.0) + ALPHA
            word_p_pos = word_p_pos / (pos_count + (len(positive_class_word_counter)+1)*ALPHA)
            p_pos += math.log(word_p_pos)
           
            word_p_neg = word_p_neg / (neg_count + (len(negative_class_word_counter)+1)*ALPHA)
            p_neg += math.log(word_p_neg)
        if (p_pos > p_neg):
            probalities.append('pos')
        else:
            probalities.append('neg')       
    return probalities

