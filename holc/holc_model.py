import tensorflow as tf
import numpy as np
import os
os.chdir( os.environ['USERPROFILE']+'/downloads/holc')
from configs import Config
config = Config()

class holcModel():
    def __init__(self, _nsteps,  _vocab_len, _maxlen ):

        # hyper parameters
        self._vocab_len = _vocab_len

        #max tokens in a sentence
        self._maxlen = _maxlen

        self._nsteps = _nsteps

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],name="dropout")

        self.input_x = tf.placeholder(tf.int32, [None, _nsteps, _maxlen], name='input_x')

        self.input_y = tf.placeholder(tf.float32, [None, config.n_classes], name='input_y')

        self.seqlen = tf.placeholder(tf.int32 , name='sentences_lengths')

        self.oplens = tf.placeholder(tf.int32, shape=[None])

        self.embedding_placeholder = tf.placeholder(tf.float32,[_vocab_len,config.dim_word])

        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # keeping track of l2 regularization loss (optional)
        self.lambda_term = config.l2_regul

        self.l2_regul_conv = config.l2_regul_conv

        self.L = 1 # the number of attention stack layers

        self.k_top = config.top_k # the number of top-k max pooling

        self.transpose_perm = [0, 3, 2, 1]

    def KMaxPooling(self,layer,l):
        k=max(self.k_top,int(((self.L-l)/self.L)*(int(layer.shape[1]))))
        top_k = tf.nn.top_k(tf.transpose(layer, perm=self.transpose_perm),k=k, sorted=True, name=None)[0]
        return tf.transpose(top_k, perm=self.transpose_perm)

    def layer_norm(self,inputs,epsilon = 1e-8,scope="ln",reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]

            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta= tf.Variable(tf.zeros(params_shape))
            gamma = tf.Variable(tf.ones(params_shape))
            normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
            outputs = gamma * normalized + beta

        return outputs

    def multihead_attention(self,inputs, encoded_output, num_units=None,
                            num_heads=config.n_heads,
                            masking=False,
                            scope="multihead_attention", reuse=None, decoding=False):

        if decoding:
            queries, keys, values = inputs, encoded_output, encoded_output
        else:
            queries, keys, values = inputs, inputs, inputs

        with tf.variable_scope(scope, reuse=reuse):
            # Set the fall back option for num_units
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            # Linear projections
            Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
            K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
            V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

            # Split and concat
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

            # Scale
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            # Key Masking
            if masking:
                key_masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))
                key_masks = tf.tile(key_masks, [num_heads, 1])
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1]) #
                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

            # Activation
            outputs = tf.nn.softmax(outputs)

            # Query Masking
            if masking:
                query_masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
                query_masks = tf.tile(query_masks, [num_heads, 1])
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
                outputs *= query_masks # broadcasting.

            # Dropouts
            outputs = tf.layers.dropout(outputs, rate=self.dropout, training=self.is_training)

            # Weighted sum
            outputs = tf.matmul(outputs, V_) #

            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2 ) #

            # Residual connection
            outputs += queries

            # Normalize
            outputs = self.layer_norm(outputs)

        return outputs

    # Position-Wise Feed-Forward Networks
    def feedforward(self,inputs, num_units=None,scope=None):
        with tf.variable_scope(scope):
            if num_units==None:
                num_units = [4*inputs.get_shape().as_list()[-1], inputs.get_shape().as_list()[-1] ]
            hidden_layer = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu, use_bias=True)
            output_layer = tf.layers.dense(hidden_layer, num_units[1], activation=tf.nn.relu, use_bias=True)

            # residual connection
            output_layer += inputs

            # Normalize
            outputs = self.layer_norm(output_layer, scope="layer_norm")

        return outputs

    def label_smoothing(self,inputs, epsilon=0.1):
        V = inputs.get_shape().as_list()[-1] # number of channels
        return ((1-epsilon) * inputs) + (epsilon / V)

    def build(self):
        # define embeddings layer
        with tf.device('/cpu:0'), tf.name_scope('word_embeddings'):
            if config.pre_trained_embs: # pre-trained embeddings
                _word_embeddings = tf.Variable(tf.constant(0.0, shape=[self._vocab_len , config.dim_word]), trainable=config.train_embeddings, name='_word_embeddings')
                self.embedding_init = _word_embeddings.assign(self.embedding_placeholder)
            else: #random initiate embeddings
                _word_embeddings = tf.Variable(tf.random_uniform([self._vocab_len+1 , config.dim_word], -1.0, 1.0),name='_word_embeddings')

        def sentence_embedding(sent_x):
            # create the sentences embeddings
            with tf.variable_scope("sentence_encoder"):
                # word embeddings
                word_embeddings =  tf.layers.dropout(tf.nn.embedding_lookup(_word_embeddings,
                    sent_x, name="sent_word_embeddings"), training=self.is_training)

                embedded_chars = tf.expand_dims(word_embeddings,-1)

                pooled_outputs_conv = []
                for i, filter_size in enumerate(config.filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # convolution Layer
                        filter_shape = [filter_size, config.dim_word, 1, config.num_feature_maps]
                        self.W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1),  name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[config.num_feature_maps]), name= "b")
                        conv  = tf.nn.conv2d(embedded_chars,self.W,strides=[1, 1,config.dim_word, 1],padding="SAME",name="conv")
                        h = tf.nn.relu(tf.nn.bias_add(conv, b) , name="relu")
                        pooled_kmax_ = self.KMaxPooling(h,self.L)

                    pooled_outputs_conv.append(pooled_kmax_)

                num_feature_maps_conv_1 = config.num_feature_maps * len(pooled_outputs_conv)* pooled_outputs_conv[0].get_shape().as_list()[1]

                h_pool_conv_1 = tf.concat(pooled_outputs_conv, 3)
                h_pool_flat  = tf.reshape(h_pool_conv_1, [-1, num_feature_maps_conv_1])

                outputs = tf.layers.dropout(h_pool_flat, rate=self.dropout, training=self.is_training)
                outputs = self.layer_norm(outputs)
            return outputs

        # create bi-direction rnn network
        def birnn(x):
            # define lstm cells with tensorflow
            # forward direction cell
            with tf.variable_scope('blstm_sentences', initializer=tf.contrib.layers.xavier_initializer()):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(config.n_hidden, forget_bias=1,name='basic_lstm_cell')
                # backward direction cell
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(config.n_hidden, forget_bias=1,name='basic_lstm_cell')

                try:
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, tf.reshape(tf.expand_dims(x, axis = 3),[len(x),-1, int((x[1].shape)[1])]),self.oplens, dtype=tf.float32,time_major=True,scope="bi_lstm")
                    outputs= tf.unstack(tf.concat(outputs,2),axis=0)

                except Exception: # old version static states
                    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)

                # provide rnn training information
                tf.summary.histogram('brnn_outputs', outputs)

                outputs = tf.layers.dropout(outputs, rate=self.dropout, training=self.is_training)

            return outputs

        # store the sentences embeddings
        _att_stack = []
        sentences = tf.unstack(self.input_x,self._nsteps,1)
        for i in range(self._nsteps):
            _att_stack.append(sentence_embedding(sentences[i]))

        # tuple the sentences embeddings
        with tf.name_scope('sentences_embeddings'):
            sent_embeddings = tf.tuple(_att_stack)

        # create the blstm layer
        with tf.variable_scope('birnn_layer'):
            blstm_outputs = birnn(sent_embeddings)

        #multi head attention
        for i in range(config.n_stacks):
            with tf.variable_scope("att_blocks_{}".format(i),reuse=tf.AUTO_REUSE):
                # Multihead Attention
                blstm_outputs = self.multihead_attention(blstm_outputs,encoded_output=None,
                decoding=False, masking=False, scope="self_att_enc")
                blstm_outputs = self.feedforward(blstm_outputs,scope='enc_pffn')

        # create the classical layer
        with tf.variable_scope('classical_layer') as scope:
            self.window = 1 if (int(self._nsteps*config.balancing_factor)==0) else int(self._nsteps*config.balancing_factor)
            # define weights for seq to seq
            self.weights_f_w=[]
            for i in range(self.window):
                self.weights_f_w.append(tf.Variable(tf.random_normal([config.n_hidden, config.n_classes])))

            self.weights_b_w=[]
            for i in range(self.window):
                self.weights_b_w.append(tf.Variable(tf.random_normal([config.n_hidden, config.n_classes])))

            self.biases_f_w =[]
            for i in range(self.window):
                self.biases_f_w.append(tf.Variable(tf.random_normal([config.n_classes])))

            self.biases_b_w =[]
            for i in range(self.window):
                self.biases_b_w.append(tf.Variable(tf.random_normal([config.n_classes])))

            _classical_layer = tf.concat([tf.matmul(tf.slice(blstm_outputs[-1-i], [0, 0], [-1, config.n_hidden]),self.weights_f_w[-1-i]) + self.biases_f_w[-1-i] for i in range(self.window)]  + [tf.matmul(tf.slice(blstm_outputs[i], [0, config.n_hidden], [-1,config.n_hidden]),self.weights_b_w[i])+ self.biases_b_w[i] for i in range(self.window)],1)

        with tf.variable_scope("prediction_scores"):
            self.pred = tf.contrib.layers.fully_connected(
            _classical_layer,
            config.n_classes,
            activation_fn=None,
            normalizer_fn=None,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),weights_regularizer=tf.contrib.layers.l2_regularizer(config.l2_regul),
            biases_initializer=tf.zeros_initializer(),
            trainable=True,
            scope="prediction_layer_scores")

        #calculate the prediction layer
        with tf.variable_scope('prediction') as scope:
            self.logits = tf.argmax(tf.nn.softmax(self.pred),1)

        with tf.name_scope('loss'):
            # define loss
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.pred, labels= (self.input_y))) # self.label_smoothing
            self.loss = tf.reduce_mean(self.loss +
            self.l2_regul_conv * tf.nn.l2_loss(self.W)+
            self.lambda_term * tf.nn.l2_loss(self.weights_f_w) +
            self.lambda_term * tf.nn.l2_loss(self.weights_b_w))

            # provide accuracy information
            tf.summary.scalar('loss', self.loss)

        with tf.name_scope('accuracy'):
            # evaluate model
            correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='num_correct')

            # provide accuracy information
            tf.summary.scalar('accuracy', self.accuracy)







