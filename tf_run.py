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



# def main():
datatest = TextDataset()
tfdx, tfdy, lx, ly = datatest.to_tf_train_input()
# tftest, tftestl = datatest.to_tf_test_input()
# it = iter(tfdx)
# print(' '.join(datatest.vocab.to_tokens(next(it).tolist())))
nput_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(tfdx, tfdy, test_size=0.2)
# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

# datatest = TextDataset()
# tfdx, tfdy, lx, ly = datatest.to_tf_train_input()
# tftest, tftestl = datatest.to_tf_test_input()
# it = iter(tftest)
# print(' '.join(datatest.vocab.to_tokens(next(it).tolist())))