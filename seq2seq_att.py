#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow_addons.seq2seq import BahdanauAttention as tfaBAttention

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io


# encoder input shape
# encoder output shape *(batch_size, max_length, hidden_size)*
# encoder hidden state shape *(batch_size, hidden_size)*


class Encoder(tf.keras.Model):
    ''' Encoder(vocab_size, embedding_dim, enc_units, batch_sz)
        encoder input shape (batch_size, max_sequence_length)
        encoder output shape (batch_size, max_length, hidden_size)
        encoder hidden state shape (batch_size, hidden_size)'''
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, attention='BahdanauAttention'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        if attention == 'BahdanauAttention':
            self.attention = BahdanauAttention(self.dec_units)
        elif attention == 'tfaBAttention':
            self.attention = tfaBAttention(self.dec_units)
        else: raise NotImplementedError

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


# * FC = Fully connected (dense) layer
# * EO = Encoder output
# * H = hidden state
# * X = input to the decoder
# * `score = FC(tanh(FC(EO) + FC(H)))`
# * `attention weights = softmax(score, axis = 1)`. Softmax by default is applied on the last axis but here we want to apply it on the *1st axis*, since the shape of score is *(batch_size, max_length, hidden_size)*. 
# * `context vector = sum(attention weights * EO, axis = 1)`. Same reason as above for choosing axis as 1.
# * `embedding output` = The input to the decoder X is passed through an embedding layer.
# * `merged vector = concat(embedding output, context vector)`
# * This merged vector is then given to the GRU

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

#%%
# BATCH_SIZE = 64
# embedding_dim = 256
# enc_units = 1024
# encoder = Encoder(10000, embedding_dim, enc_units, BATCH_SIZE)

# %%
