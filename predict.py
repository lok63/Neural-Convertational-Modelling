import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np
import config



data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)

def predict_ids_to_seq(predict_ids, id2word, beam_szie,beam_search):
    '''
    Convert the result returned by beam_search to a string
    :param predict_ids: list, length is batch_size, each element is an array of decode_len*beam_size
    :param id2word: vocab dictionary
    :return:
    ''' 
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))
            if not beam_search :break

with tf.Session() as sess:
    model = Seq2SeqModel(config.RNN_SIZE, config.NUM_LAYERS, config.EMBDEDDING_SIZE, config.LR, word2id,
                         mode='decode', use_attention=config.USE_ATTENTION, beam_search=config.BEAM_SEARCH, beam_size=config.BEAM_SIZE, max_gradient_norm=config.MAX_GRAD_NORM)
    ckpt = tf.train.get_checkpoint_state(config.MODEL_DIR)
    if ckpt :
        print('Reloading model parameters..')
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No such file:[{}]'.format(config.MODEL_DIR))
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        batch = sentence2enco(sentence, word2id)
        predicted_ids = model.infer(sess, batch)
        # print(predicted_ids)
        predict_ids_to_seq(predicted_ids, id2word, model.beam_size, model.beam_search)
        print("> ", "")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
