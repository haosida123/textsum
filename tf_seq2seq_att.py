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
# from data import MyCorpus
# from utils.config import data_path


class Seq2seq_attention():
    def __init__(self, vocab_size, params, embedding_matrix=None):
        params = self.params = params.from_json()  # update if json file changed
        encoder = Encoder(vocab_size, params["embedding_dim"],
                          params["enc_dec_units"], params["batch_size"], embedding_matrix)
        decoder = Decoder(
            vocab_size, params["embedding_dim"], params["enc_dec_units"], params["batch_size"], params["att_units"], embedding_matrix=embedding_matrix, activation=params["activation"])
        self.encoder = encoder
        self.decoder = decoder

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, targ, enc_hidden, begin_id):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims(
                [begin_id] * self.params["batch_size"], 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output)
                if self.my_loss_function is not None:
                    loss += self.my_loss_function(targ[:, t], predictions)
                else:
                    loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train_epoch(
        self, dataset, epochs, steps_per_epoch, begin_id, optimizer=None,
        loss_function=None, restore_checkpoint=False, callback=None,
        datatest=None
    ):
        self.optimizer = tf.keras.optimizers.Adam() if optimizer is None else optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none') if loss_function is None else None
        self.my_loss_function = loss_function
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.checkpoint_dir = os.path.join(self.params.data_path, 'training_checkpoints')
        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        if restore_checkpoint:
            print('restoring checkpoint')
            try:
                self.restore_checkpoint()
            except Exception as e:
                print(e)
                if input('continue without restoring?(y/n)').lower() != 'y':
                    return
        print('start total {} epoch(s), {} steps per epoch, batch size {}'.format(
            epochs, steps_per_epoch, self.params["batch_size"]))
        for epoch in range(epochs):
            start = time.time()
            if callback is not None:
                callback()
            enc_hidden = self.encoder.initialize_hidden_state(self.params["batch_size"])
            if datatest is not None:
                print('test data loss:')
                print(self.teacher_forcing_test_loss(datatest[0], datatest[1], begin_id).numpy())
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(
                    inp, targ, enc_hidden, begin_id)
                total_loss += batch_loss
                if batch % (steps_per_epoch // 50 + 1) == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Time {}'.format(
                        epoch + 1, batch, batch_loss.numpy(), time.time() - start))
            # saving (checkpoint) the model every 2 epochs
            # if (epoch + 1) % 2 == 0:
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
            self.checkpoint_dir = os.path.join(self.params.data_path, 'training_checkpoints')
            if not hasattr(self, "optimizer"):
                self.checkpoint = tf.train.Checkpoint(
                    encoder=self.encoder, decoder=self.decoder)
            else:
                self.checkpoint = tf.train.Checkpoint(
                    optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    def compare_input_output(self, input_array, vocab, max_tar_length, targets=None):
        """input_array: array representing single sentence"""
        predict_array_input(
            input_array.numpy().tolist(), vocab, max_tar_length,
            self.encoder, self.decoder, targets)

    def teacher_forcing_test_loss(self, test_x, test_y, begin_id):
        """same as train loss calculating method"""
        loss = 0
        enc_hidden = tf.zeros((test_x.shape[0], self.encoder.enc_dec_units))
        enc_output, enc_hidden = self.encoder(test_x, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims(
            [begin_id] * test_x.shape[0], 1)
        # Teacher forcing - feeding the target as the next input
        for t in range(1, test_y.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = self.decoder(
                dec_input, dec_hidden, enc_output)
            if self.my_loss_function is not None:
                loss += self.my_loss_function(test_y[:, t], predictions)
            else:
                loss += self.loss_function(test_y[:, t], predictions)
            # using teacher forcing
            dec_input = tf.expand_dims(test_y[:, t], 1)
        batch_loss = (loss / int(test_y.shape[1]))
        return batch_loss


class Encoder(tf.keras.Model):
    ''' Encoder(vocab_size, embedding_dim, enc_dec_units, batch_sz)
        encoder input shape (batch_size, max_sequence_length)
        encoder output shape (batch_size, max_length, hidden_size)
        encoder hidden state shape (batch_size, hidden_size)'''

    def __init__(self, vocab_size, embedding_dim, enc_dec_units, batch_sz, embedding_matrix=None):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_dec_units = enc_dec_units
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(
                    embedding_matrix),
                trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.enc_dec_units))


class Decoder(tf.keras.Model):
    """(self, vocab_size, embedding_dim, enc_dec_units, batch_sz, attention_units, attention='BahdanauAttention', embedding_matrix=None)"""
    def __init__(self, vocab_size, embedding_dim, enc_dec_units, batch_sz, attention_units, attention='BahdanauAttention', embedding_matrix=None,
    activation='softmax'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_dec_units = enc_dec_units
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(
                    embedding_matrix),
                trainable=False)
            self.fc1 = tf.keras.layers.Dense(vocab_size, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(embedding_matrix.transpose()), activation=activation, trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
            self.fc1 = tf.keras.layers.Dense(vocab_size, use_bias=False, activation=activation)
        self.gru = tf.keras.layers.GRU(self.enc_dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc0 = tf.keras.layers.Dense(embedding_dim)
        # used for attention
        if attention == 'BahdanauAttention':
            self.attention = BahdanauAttention(attention_units)  # self.enc_dec_units)
        elif attention == 'tfaBAttention':
            self.attention = tfaBAttention(attention_units)  # self.enc_dec_units)
        else:
            raise NotImplementedError

    def call(self, x, hidden, enc_output):
        """
        x shape: (batch_size, 1)
        hidden: (batch size, enc units)
        enc_output: (batch size, sequence length, enc units)
        output: (batch, vocab)
        state: (batch, dec units)
        """
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # x shape before: (batch_size, 1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab)
        x = self.fc1(self.fc0(output))
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
        # hidden shape (query) == (batch_size, hidden size)
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


def fasttext_embedding(params, load_file=True, sentences=None):
    """sentences: Corpus object. returns embedding matrix"""
    file = os.path.join(params.data_path, "fasttext_embedding.npy")
    # file = os.path.join(data_path, "modelfasttext.model")
    if load_file:
        try:
            # modelfasttext = gensim.models.fasttext.FastText.load(file)
            return np.load(file, allow_pickle=False)
        except FileNotFoundError:
            load_file = False
    if not load_file:
        if sentences is None:
            raise (RuntimeError, 'arg sentences should be a generator of sentences')
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # sentences = MyCorpus()
        modelfasttext = gensim.models.fasttext.FastText(sentences=sentences,
                                                        size=params.embedding_dim, window=5, min_count=params.vocab_min_frequency, workers=3)
        # modelfasttext.save(file)
        embedding_matrix = np.array(
            [modelfasttext.wv[token] for token in sentences.vocab.idx_to_token])
        np.save(file, embedding_matrix, allow_pickle=False)
    return embedding_matrix  # , sentences.vocab


def layer_info(layer):
    name_shapes = [(v.name, v.shape, reduce(lambda x, y: x * y, v.shape)) for v in layer.trainable_variables]
    print(
        '\n'.join(['; '.join(
            [nss[0], str(nss[1]), '{:,}'.format(
                reduce(lambda x, y: x * y, nss[1]))]) for nss in name_shapes]))
    print('total trainable weights: {:,}'.format(sum([ns[2] for ns in name_shapes])))
    return
# def load_encoder_decoder(vocab_size, params, checkpoint_dir=None, optimizer=None, embedding_matrix=None):
#     """
# Returns
# -------
# encoder

# decoder
# """
#     encoder = Encoder(vocab_size, params["embedding_dim"], params["enc_dec_units"], params["batch_size"], embedding_matrix)
#     decoder = Decoder(vocab_size, params["embedding_dim"], params["enc_dec_units"], params["batch_size"])
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
