# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import tensorflow as tf
import unicodedata
import re
import numpy as np
import os
import io
import time

from data import TextDataset

from utils.config import data_path
from seq2seq_att import BahdanauAttention, Encoder, Decoder
from tf_train import train_epochs
from tf_eval import predict



# def main():
datatext = TextDataset()
tfdx, tfdy, seq_length_x, seq_length_y = datatext.to_tf_train_input()

vocab_size = len(datatext.vocab.idx_to_token)
# tftest, tftestl = datatext.to_tf_test_input()
# it = iter(tfdx)
# print(' '.join(datatext.vocab.to_tokens(next(it).tolist())))
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(tfdx, tfdy, test_size=0.2)
# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
#%%
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

#%%

encoder = Encoder(vocab_size, embedding_dim, units, BATCH_SIZE)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

attention_layer = BahdanauAttention(64)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

decoder = Decoder(vocab_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

#%%
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

"""## Checkpoints (Object-based saving)"""
checkpoint_dir = os.path.join(data_path, 'training_checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)
train_epochs()
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
predict("奔驰")
# datatest = TextDataset()
# tfdx, tfdy, lx, ly = datatest.to_tf_train_input()
# tftest, tftestl = datatest.to_tf_test_input()
# it = iter(tftest)
# print(' '.join(datatest.vocab.to_tokens(next(it).tolist())))