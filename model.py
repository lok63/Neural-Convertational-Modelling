import tensorflow as tf
import os
from tensorflow.python.util import nest
import config

def get_inputs():
    # Create the variables/placeholders for the model
    print('Initialising placeholder...')
    encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
    decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')

    encoder_seq_length = tf.placeholder(tf.int32, [None], name='encoder_seq_length')
    decoder_seq_length = tf.placeholder(tf.int32, [None], name='decoder_seq_length')

    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    keep_probs = tf.placeholder(tf.float32, name='keep_probs')

    max_seq_len = tf.reduce_max(decoder_seq_length, name='max_seq_len')
    mask = tf.sequence_mask(decoder_seq_length, max_seq_len, dtype=tf.float32, name='masks')

    return encoder_inputs,decoder_targets, encoder_seq_length, decoder_seq_length,batch_size,keep_probs, max_seq_len, mask

def create_rnn_cell(rnn_size,num_layers,keep_probs):
    # Initialise the RNN cell
    def single_rnn_cell(rnn_size,num_layers,keep_probs):
        # The cell will be constructed using LSTM.
        # You can change the code and use GRU cells instead
        single_cell = tf.contrib.rnn.LSTMCell(rnn_size)
        # Add dropout rate 
        cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=keep_probs)
        return cell
    cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell(rnn_size,num_layers,keep_probs) for _ in range(num_layers)])
    return cell


def encoder(encoder_inputs,rnn_size,num_layers,encoder_seq_length,keep_probs,embedding_size,vocab_size):

    with tf.name_scope('encoder'):
        #encoder cell using LSTM and dropout
        encoder_cell = create_rnn_cell(rnn_size,num_layers,keep_probs)
        # Embeddings are dense vector representations of the characters in the sequence.
        # In this case the vector will be desribed using the word IDs
        embedding = tf.get_variable('embedding', [vocab_size, embedding_size])
        encoder_inputs_embedded = tf.nn.embedding_lookup(embedding,encoder_inputs)

        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell, 
                                                            encoder_inputs_embedded,
                                                           sequence_length=encoder_seq_length,
                                                           dtype=tf.float32)
        return encoder_outputs, encoder_states,embedding


def decoder(encoder_outputs, encoder_states,encoder_seq_length,decoder_targets,decoder_seq_length,decoder_cell,decoder_initial_state,
                                                        embedding,vocab_size,batch_size,word_to_idx,max_seq_len,beam_search,beam_size,mode):
    print("Building Decoder")
    with tf.name_scope('decoder'):
        # When beam search is used we have to ensure that:
        # *The encoder output has been tiled to beam_width
        # *The batch_size argument passed to the zero_state
        # *The initial state created with zero_state above contains a cell_state value containing properly tiled final state from the encoder.
        if config.BEAM_SEARCH:
            print("use beamsearch decoding..")
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_size)
            encoder_states = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, beam_size), encoder_states)
            encoder_seq_length = tf.contrib.seq2seq.tile_batch(encoder_seq_length, multiplier=beam_size)


        output_layer = tf.layers.Dense(vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        if mode == 'train':
            # The inputs have to be pre-processed before we use them in the decoding part of the model
            ending = tf.strided_slice(decoder_targets, [0, 0], [batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([batch_size, 1], word_to_idx['<go>']), ending], 1)
            decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
            # A helper function is used for passing the elemennt in the Basic Decoder
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                sequence_length=decoder_seq_length,
                                                                time_major=False, 
                                                                name='training_helper')
            
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=training_helper,
                                                               initial_state=decoder_initial_state, output_layer=output_layer)
            
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                      impute_finished=True,
                                                                      maximum_iterations=max_seq_len)
            return decoder_outputs, _


         #For inference 
        elif mode == 'decode':
            start_tokens = tf.ones([batch_size, ], tf.int32) * word_to_idx['<go>']
            end_token = word_to_idx['<eos>']

            # The decoder stage determines different combinations depending on whether or not beam_search is used.
            # BeamSearchDecoder (which already implements the helper class) if it is used.
            # If not used, call GreedyEmbeddingHelper+BasicDecoder combination for greedy decoding
            if config.BEAM_SEARCH:
                inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, 
                                                                         embedding=embedding,
                                                                         start_tokens=start_tokens, 
                                                                         end_token=end_token,
                                                                         initial_state=decoder_initial_state,
                                                                         beam_width=beam_size,
                                                                         output_layer=output_layer)
            else:
                decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                           start_tokens=start_tokens, 
                                                                           end_token=end_token)

                inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                                                    helper=decoding_helper,
                                                                    initial_state=decoder_initial_state,
                                                                    output_layer=output_layer)

            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                                      maximum_iterations=10)
            # Call dynamic_decode for decoding, decoder_outputs is a namedtuple,
            # For beam_search, it contains two items (rnn_outputs, sample_id)
            # rnn_output: [batch_size, decoder_targets_length, vocab_size]
            # sample_id: [batch_size, decoder_targets_length], tf.int32

            # When using beam_search, it contains two items (predicted_ids, beam_search_decoder_output)
            # predicted_ids: [batch_size, decoder_targets_length, beam_size],Save output
            # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
            # So the corresponding only needs to return predicted_ids or sample_id to translate into the final result.
            if config.BEAM_SEARCH:
                decoder_predict_decode = decoder_outputs.predicted_ids
            else:
                decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)

            return decoder_outputs, decoder_predict_decode

def create_context(encoder_outputs,encoder_states,rnn_size,num_layers,encoder_seq_length,keep_probs,batch_size,use_attention,beam_search,beam_size):
    # This function is used to develop the context vector that will benefit the decoder
    # If attention is not used, then the vanilla method will use the encoder stated as the default context vector
    print("Creating Context...")
    print("Using Attention") if config.USE_ATTENTION else print("Using Vanilla")

    # When beam search is used we have to ensure that:
    # *The encoder output has been tiled to beam_width
    # *The batch_size argument passed to the zero_state
    # *The initial state created with zero_state above contains a cell_state value containing properly tiled final state from the encoder.
    if config.BEAM_SEARCH:
        encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_size)
        encoder_states = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, beam_size), encoder_states)
        encoder_seq_length = tf.contrib.seq2seq.tile_batch(encoder_seq_length, multiplier=beam_size)

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                               memory=encoder_outputs,
                                                               memory_sequence_length=encoder_seq_length)
    decoder_cell = create_rnn_cell(rnn_size,num_layers,keep_probs)

    if config.USE_ATTENTION:
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, 
                                                          attention_mechanism=attention_mechanism,
                                                          attention_layer_size=rnn_size, 
                                                          name='Attention_Wrapper')
                                                       
        batch_size = batch_size if not beam_search else batch_size * beam_size
        decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_states)
    else:
        decoder_initial_state = encoder_states

    return decoder_cell,decoder_initial_state

def opt(decoder_outputs,decoder_targets,mask,learning_rate,max_gradient_norm):
    # Optimisation process which will use gtadient normalisation, gradient clipping 
    with tf.name_scope('optimizer'):
        decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
        decoder_predict_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_pred_train')
        # Use sequence_loss to calculate loss, here you need to pass in the previously defined mask flag
        loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits_train,
                                                targets=decoder_targets, 
                                                weights=mask)

        # Training summary for the current batch_loss
        optimizer = tf.train.AdamOptimizer(learning_rate)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        # Training summary for the current batch_loss
        # for i,grad in enumerate(gradients):
        #     tf.summary.histogram(str(i)+ '/gradient', grad)

        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()

        return train_op,loss,summary_op,decoder_predict_train


class Seq2SeqModel(object):
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, mode, use_attention,
         beam_search, beam_size, max_gradient_norm):


        vocab_size = len(word_to_idx)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.beam_search = beam_search
        self.beam_size =beam_size
        self.encoder_inputs,self.decoder_targets, self.encoder_seq_length, self.decoder_seq_length,self.batch_size,self.keep_probs, self.max_seq_len, self.mask = get_inputs()


        enc_outputs, enc_states,embedding = encoder(self.encoder_inputs, 
                              rnn_size,
                              num_layers, 
                              self.encoder_seq_length, 
                              self.keep_probs, 
                              embedding_size, 
                              vocab_size)




        decoder_cell, decoder_initial_state = create_context( 
                                              enc_outputs, 
                                              enc_states,
                                              rnn_size,
                                              num_layers, 
                                              self.encoder_seq_length, 
                                              self.keep_probs, 
                                              self.batch_size,
                                              use_attention,
                                              beam_search,
                                              beam_size)
        

        self.decoder_outputs,self.decoder_predict_decode = decoder(enc_outputs, 
                                                                      enc_states, 
                                                                      self.encoder_seq_length,
                                                                      self.decoder_targets,
                                                                      self.decoder_seq_length,
                                                                      decoder_cell,
                                                                      decoder_initial_state,
                                                                      embedding,
                                                                      vocab_size,
                                                                      self.batch_size,
                                                                      word_to_idx,
                                                                      self.max_seq_len,
                                                                      beam_search,
                                                                      beam_size,
                                                                      mode)

        if mode == 'train':
            self.train_op, self.loss, self.summary_op, self.decoder_predict_train = opt(self.decoder_outputs, 
                                                                                       self.decoder_targets, 
                                                                                       self.mask,
                                                                                       learning_rate,
                                                                                       max_gradient_norm)
        self.saver = tf.train.Saver(tf.global_variables())



    def train(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_seq_length: batch.encoder_seq_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_seq_length: batch.decoder_seq_length,
                      self.keep_probs: 0.5,
                      self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary, pred, target = sess.run([self.train_op, self.loss, self.summary_op, self.decoder_predict_train, self.decoder_targets], feed_dict=feed_dict)
        return loss, summary,pred,target



    def infer(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_seq_length: batch.encoder_seq_length,
                      self.keep_probs: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict


