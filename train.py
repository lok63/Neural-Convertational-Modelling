import tensorflow as tf
from data_helpers import loadDataset, getBatches, sentence2enco
from nltk.translate.bleu_score import sentence_bleu
from model import Seq2SeqModel
from tqdm import tqdm
import config
import math
import os
conf_gpu = tf.ConfigProto()
conf_gpu.gpu_options.allow_growth = True
import warnings
warnings.filterwarnings("ignore")


def get_bleu(reference,predicted):
    #Convert pred to the right datatype 

    mean_score=[]

    for i in range(config.BATCH_SIZE):
        bleu_4= sentence_bleu([reference[i]], predicted[i], weights=(0.25, 0.25, 0.25, 0.25))
        mean_score.append(bleu_4)

    return sum(mean_score) / len(mean_score) 

def _check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('model' + '/checkpoint'))

    if ckpt and ckpt.model_checkpoint_path:
        print("Loading parameters for the Chatbot")
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initializing fresh parameters for the Chatbot")


data_path = 'data/dataset-cornell-length10-filter1-vocabSize40000.pkl'
word2id, id2word, trainingSamples = loadDataset(data_path)


with tf.Session(config = conf_gpu) as sess:
    model = Seq2SeqModel(config.RNN_SIZE, config.NUM_LAYERS, config.EMBDEDDING_SIZE, config.LR, word2id,
                         mode='train', use_attention=config.USE_ATTENTION, beam_search=config.BEAM_SEARCH, beam_size=config.BEAM_SIZE, max_gradient_norm=config.MAX_GRAD_NORM)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.MODEL_DIR + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        print('Reloading model parameters..')
        saver.restore(sess, ckpt.model_checkpoint_path)
        current_step  = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        print(current_step)
    else:
        print('Created new model parameters..')
        current_step = 0

    total_loss = 0
    total_bleu = 0
    total_perpl = 0

    summary_writer = tf.summary.FileWriter(config.LOGS, graph=sess.graph)
    for e in range(config.EPOCHS):
        print("----- Epoch {}/{} -----".format(e + 1, config.EPOCHS))
        batches = getBatches(trainingSamples, config.BATCH_SIZE)
        if e!= 0:
            checkpoint_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
            saver.save(sess, checkpoint_path, e)

        for nextBatch in tqdm(batches, desc="Training"):
            if current_step == 150000:
                break
            else:
                loss, summary,pred, target = model.train(sess, nextBatch)
                bleu = get_bleu(target, pred)
                total_loss += loss
                total_perpl += 2**(float(loss))
                total_bleu += bleu

                current_step += 1

                if current_step % config.STEPS == 0:
                    perplexity = 2**(float(loss)) if loss < 300 else float('inf')
                    tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f -- BLEU %.3f"  % (current_step, total_loss/config.STEPS, total_perpl/config.STEPS, total_bleu/config.STEPS))

                    total_loss = 0
                    total_bleu = 0
                    total_perpl = 0
                    

                    summary_writer.add_summary(summary, current_step)
                    checkpoint_path = os.path.join(config.MODEL_DIR, config.MODEL_NAME)
                    saver.save(sess, checkpoint_path, global_step=current_step)
