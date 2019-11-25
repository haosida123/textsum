# %%
from sklearn.model_selection import train_test_split

import tensorflow as tf

from data import TextDataset

from utils.config import params
from tf_seq2seq_att import fasttext_embedding, Seq2seq_attention
# from tf_train import train_epochs
# from tf_eval import predict_array_input
# from tf_train import fasttext_embedding



# def main():
datatext = TextDataset(data_size=100)
# datatext = TextDataset()
tfdx, tfdy, seq_length_x, seq_length_y = datatext.to_tf_train_input()

vocab_size = len(datatext.vocab)
# tftest, tftestl = datatext.to_tf_test_input()
# it = iter(tfdx)
# print(' '.join(datatext.vocab.to_tokens(next(it).tolist())))
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(tfdx, tfdy, test_size=0.2)
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
#%%
BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train)//params.batch_size

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(params.batch_size, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape


#%%
seq2seq = Seq2seq_attention(vocab_size, params)
# seq2seq = Seq2seq_attention(vocab_size, params, embedding_matrix=fasttext_embedding(params))
seq2seq.compare_input_output(example_input_batch[0], datatext, max(seq_length_y), example_target_batch[0])
# predict_array_input(example_input_batch[0].numpy().tolist(), datatext.vocab, max(seq_length_y), encoder, decoder)
#%%
seq2seq.weight_info()
#%%
seq2seq.train_epoch(dataset, 2, steps_per_epoch,datatext.vocab.bos, restore_checkpoint=False)
seq2seq.compare_input_output(example_input_batch[0], datatext, max(seq_length_y), example_target_batch[0])



# %%
