from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import nltk
import numpy as np
import pickle
import random

padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    #Batch class, which contains the encoder input, decoder input, decoder tag, decoder sample length mask
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_seq_length = []
        self.decoder_targets = []
        self.decoder_seq_length = []

# def loadDataset(filename):

#     dataset_path = os.path.join(filename)
#     print('Loading dataset from {}'.format(dataset_path))
#     data = open(dataset_path, "rb")
#     word2id = pickle.load(data)
#     id2word = pickle.load(data)
#     trainingSamples = pickle.load(data)
#     data.close()
#     return word2id, id2word, trainingSamples

def loadDataset(filename):
    '''
    read sample data
    :param filename: The file path is a dictionary containing word2id and id2word, respectively a dictionary and a reverse dictionary corresponding to the word and the index
    :return: word2id, id2word, trainingSamples
    '''
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)  # Warning: If adding something here, also modify saveDataset
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    return word2id, id2word, trainingSamples

def createBatch(samples):
    '''
     According to the given samples (that is, a batch of data), padding and constructing the data form required by the placeholder
    :param samples: sample data of a batch, list, each element is in the form of [question, answer], id
    :return: can directly pass the data format of feed_dict after processing
    '''
    batch = Batch()
    batch.encoder_seq_length = [len(sample[0]) for sample in samples]
    batch.decoder_seq_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_seq_length)
    max_target_length = max(batch.decoder_seq_length)

    for sample in samples:
        #The source is reversed and the maximum length of the PAD value is the batch.
        source = list(reversed(sample[0]))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

        #Target the PAD and add the END symbol
        target = sample[1]
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)
        #batch.target_inputs.append([goToken] + target + pad[:-1])

    return batch

def getBatches(data, batch_size):
    '''
    The raw data is divided into different small batches based on all the data read out and batch_size. Call the createBatch function on each batch index sample.
    :param data: The trainingSamples after the loadDataset function is read, which is the list of QA pairs.
    :param batch_size: batch size
    :param en_de_seq_len: list, the first element represents the maximum length of the source end sequence, and the second element represents the maximum length of the target end sequence
    :return: list, each element is a batch of sample data, can be directly passed to feed_dict for training
    '''
    #samples are shuffled before each epoch
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches

def sentence2enco(sentence, word2id):
    '''
    When testing, the user-entered sentences are converted into data that can be directly fed into the model. Now the sentence is converted to id, and then call to createBatch.
    :param sentence: the sentence entered by the user
    :param word2id: dictionary of correspondence between words and id
    :param en_de_seq_len: list, the first element represents the maximum length of the source end sequence, and the second element represents the maximum length of the target end sequence
    :return: The processed data can be directly fed into the model for prediction
    '''
    if sentence == '':
        return None
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) > 20:
        return None
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    batch = createBatch([[wordIds, []]])
    return batch
