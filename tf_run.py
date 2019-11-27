# %%
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data import TextDataset  #, MyCorpus
from utils.config import params
from tf_seq2seq_att import fasttext_embedding, Seq2seq_attention
from beam_search import BeamSearch
# from rouge_l_tensorflow import tf_rouge_l

print(params.__dict__)
datatext = TextDataset()
tfdx, tfdy, seq_length_x, seq_length_y = datatext.to_tf_train_input()

vocab_size = len(datatext.vocab)
# tftest, tftestl = datatext.to_tf_test_input()
# it = iter(tfdx)
# print(' '.join(datatext.vocab.to_tokens(next(it).tolist())))
#%%
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    tfdx, tfdy, test_size=0.1)
print(len(input_tensor_train), len(target_tensor_train),
      len(input_tensor_val), len(target_tensor_val))
BUFFER_SIZE = len(input_tensor_train)
steps_per_epoch = len(input_tensor_train)//params.batch_size

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(params.batch_size, drop_remainder=True)


# %%
# seq2seq = Seq2seq_attention(vocab_size, params)
seq2seq = Seq2seq_attention(
    vocab_size, params, embedding_matrix=fasttext_embedding(params, sentences=None))
seq2seq.encoder.embedding.trainable = False
seq2seq.decoder.embedding.trainable = False
seq2seq.decoder.fc1.trainable = False
it = iter(dataset.unbatch())
inp, out = next(it)
beam_search = BeamSearch(seq2seq, 9, datatext.vocab.bos, datatext.vocab.eos, max(seq_length_y))
seq2seq.compare_input_output(inp, datatext.vocab, max(seq_length_y), out, beam_search)
seq2seq.weight_info()

# %%


# def my_loss(truth, preds):
#     return sum(tf_rouge_l(preds, truth, datatext.vocab.eos))
# seq2seq.train_epoch(dataset, 5, steps_per_epoch,datatext.vocab.bos, loss_function=my_loss, restore_checkpoint=True, datatest=(input_tensor_val[:1], target_tensor_val[:1]))
def callback():
    seq2seq.compare_input_output(inp, datatext.vocab, max(seq_length_y), out, beam_search)

seq2seq.train_epoch(dataset, 5, steps_per_epoch, datatext.vocab.bos,
                    restore_checkpoint=True, datatest=(input_tensor_val[:100], target_tensor_val[:100]), callback=callback)
seq2seq.compare_input_output(inp, datatext.vocab, max(seq_length_y), out, beam_search)


# %%
