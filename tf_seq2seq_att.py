from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow_addons.seq2seq import BahdanauAttention as tfaBAttention
import time
import os
import logging
import gensim
import numpy as np
from functools import reduce

from tf_eval import predict_array_input
from data import MyCorpus
from utils.config import data_path


class Seq2seq_attention():
    def __init__(self, vocab_size, params, embedding_matrix=None):
        self.params = params
        encoder = Encoder(vocab_size, params["embedding_dim"],
                          params["enc_units"], params["batch_size"], embedding_matrix)
        decoder = Decoder(
            vocab_size, params["embedding_dim"], params["dec_units"], params["batch_size"])
        self.encoder = encoder
        self.decoder = decoder

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, targ, enc_hidden, vocab_beginning_token_index):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims(
                [vocab_beginning_token_index] * self.params["batch_size"], 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train_epoch(
        self, dataset, epochs, steps_per_epoch, vocab_beginning_token_index, optimizer=None,
        loss_object=None, restore_checkpoint=False
    ):
        self.optimizer = tf.keras.optimizers.Adam() if optimizer is None else optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none') if loss_object is None else loss_object
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.checkpoint_dir = os.path.join(data_path, 'training_checkpoints')
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        if restore_checkpoint:
            print('restoring checkpoint')
            self.restore_checkpoint()

        print('start total {} epoch(s), {} steps per epoch, batch size {}'.format(
            epochs, steps_per_epoch, self.params["batch_size"]))
        for epoch in range(epochs):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(
                    inp, targ, enc_hidden, vocab_beginning_token_index)
                total_loss += batch_loss
                if batch % (steps_per_epoch // 10 + 1) == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(
                        epoch + 1, batch, batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=checkpoint_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def weight_info(self):
        name_shapes = [(v.name, v.shape, reduce(lambda x, y: x * y, v.shape)) for v in (
            self.encoder.trainable_variables + self.decoder.trainable_variables)]
        print(
            '\n'.join(['; '.join(
                [nss[0], str(nss[1]), '{:,}'.format(
                    reduce(lambda x, y: x * y, nss[1]))]) for nss in name_shapes]))
        print('total trainable weights: {:,}'.format(sum([ns[2] for ns in name_shapes])))
        return

    def restore_checkpoint(self):
        if not hasattr(self, "checkpoint"):
            if not hasattr(self, "optimizer"):
                self.checkpoint = tf.train.Checkpoint(
                    encoder=self.encoder, decoder=self.decoder)
            else:
                self.checkpoint = tf.train.Checkpoint(
                    optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    def compare_input_output(self, input_array, datatext, max_tar_length, targets=None):
        """input_array: array representing single sentence"""
        predict_array_input(
            input_array.numpy().tolist(), datatext.vocab, max_tar_length,
            self.encoder, self.decoder, targets)


def fasttext_embedding(params, load_file=True):
    """returns embedding matrix"""
    file = os.path.join(data_path, "fasttext_embedding.npy")
    # file = os.path.join(data_path, "modelfasttext.model")
    if load_file:
        try:
            # modelfasttext = gensim.models.fasttext.FastText.load(file)
            return np.load(file, allow_pickle=False)
        except FileNotFoundError:
            load_file = False
    if not load_file:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = MyCorpus()
        modelfasttext = gensim.models.fasttext.FastText(sentences=sentences,
                                                        size=params.embedding_dim, window=5, min_count=params.vocab_min_frequency, workers=4)
        # modelfasttext.save(file)
        embedding_matrix = np.array(
            [modelfasttext.wv[token] for token in sentences.vocab.idx_to_token])
        np.save(file, embedding_matrix, allow_pickle=False)
    return embedding_matrix  # , sentences.vocab


# def load_encoder_decoder(vocab_size, params, checkpoint_dir=None, optimizer=None, embedding_matrix=None):
#     """
# Returns
# -------
# encoder

# decoder
# """
#     encoder = Encoder(vocab_size, params["embedding_dim"], params["enc_units"], params["batch_size"], embedding_matrix)
#     decoder = Decoder(vocab_size, params["embedding_dim"], params["dec_units"], params["batch_size"])
#     if checkpoint_dir:
#         try:
#             if optimizer:
#                 checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
#             else:
#                 checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
#             checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
#         except Exception as e:
#             print('error', e)
#     return encoder, decoder


class Encoder(tf.keras.Model):
    ''' Encoder(vocab_size, embedding_dim, enc_units, batch_sz)
        encoder input shape (batch_size, max_sequence_length)
        encoder output shape (batch_size, max_length, hidden_size)
        encoder hidden state shape (batch_size, hidden_size)'''

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix=None):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(
                    embedding_matrix),
                trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
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
        else:
            raise NotImplementedError

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
