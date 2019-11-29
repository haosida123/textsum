
import tensorflow as tf
# import tensorflow_addons as tfa
import time
import os
import logging
import gensim
import numpy as np
from functools import reduce

from tf_eval import predict_array_input
# from data import MyCorpus
# from utils.config import data_path


class Seq2seq_attention(tf.keras.Model):
    def __init__(self, vocab_size, params, embedding_matrix=None):
        super(Seq2seq_attention, self).__init__()
        params = self.params = params.from_json()  # update if json changed
        encoder = Encoder(vocab_size, params["embedding_dim"],
                          params["enc_dec_units"], params["batch_size"], embedding_matrix)
        decoder = Decoder(
            vocab_size, params["embedding_dim"], params["enc_dec_units"], params["batch_size"], params["att_units"], attention=params["attention"], embedding_matrix=embedding_matrix, activation=params["activation"])
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
        loss = tf.cast(0, tf.float32)
        batch_size = len(inp)
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.cast(tf.expand_dims(
                [begin_id] * batch_size, 1), tf.int32)
            attention_coverage = tf.cast([0] * enc_output.shape[1], tf.float32)
            attention_coverage = tf.zeros((batch_size, enc_output.shape[1]),
                                          dtype=tf.float32)
            # Teacher forcing - feeding the target as the next input
            for t in tf.range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, att_weights = self.decoder(
                    dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:, t], predictions)
                # Coverage loss
                att_weights = tf.squeeze(att_weights, axis=2)
                attention_coverage += att_weights
                attention_coverage_loss = tf.reduce_sum(tf.reduce_min(tf.concat([
                    attention_coverage, att_weights], axis=0), axis=0))
                loss += attention_coverage_loss * self.coverage_weight
                # using teacher forcing
                dec_input = tf.cast(tf.expand_dims(targ[:, t], 1), tf.int32)
        batch_loss = (loss / int(targ.shape[1]))
        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss

    def train_epoch(
        self,
        dataset,
        epochs,
        steps_per_epoch,
        begin_id,
        coverage_weight=0.9,
        optimizer=None,
        loss_function=None,
        restore_checkpoint=False,
        callback=None,
        dataval=None
    ):
        self.coverage_weight = coverage_weight
        self.optimizer = tf.keras.optimizers.Adam() if optimizer is None else optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none') if loss_function is None else None
        if loss_function is not None:
            self.loss_function = loss_function
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.checkpoint_dir = os.path.join(
            self.params.data_path, 'training_checkpoints')
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
            enc_hidden = self.encoder.initialize_hidden_state(batch_size=self.params["batch_size"])
            total_loss = 0
            for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
                batch_loss = self.train_step(
                    inp, targ, enc_hidden, begin_id)
                total_loss += batch_loss
                if batch % (steps_per_epoch // 100 + 1) == 0:
                    print('Epoch {} Batch {} Loss {:.4f} Time {}'.format(
                        epoch + 1, batch, batch_loss.numpy(), time.time() - start))
            # saving (checkpoint) the model every 2 epochs
            # if (epoch + 1) % 2 == 0:
            self.checkpoint.save(file_prefix=checkpoint_prefix)
            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            if dataval is not None:
                # print('test data loss:')
                # print(self.teacher_forcing_test_loss(dataval, begin_id).numpy())
                self.teacher_forcing_test_loss(dataval, begin_id)
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def weight_info(self):
        print('encoder:\n', '  \t'.join(
            [layer.name + ':trainable: {}'.format(layer.trainable) for layer in self.encoder.layers]))
        print('decoder:\n', '  \t'.join(
            [layer.name + ':trainable: {}'.format(layer.trainable) for layer in self.decoder.layers]))
        name_shapes = [(v.name, v.shape, reduce(lambda x, y: x * y, v.shape)) for v in (
            self.encoder.trainable_variables + self.decoder.trainable_variables)]
        print(
            '\n'.join(['; '.join(
                [nss[0], str(nss[1]), '{:,}'.format(
                    reduce(lambda x, y: x * y, nss[1]))]) for nss in name_shapes]))
        print('total trainable weights: {:,}'.format(
            sum([ns[2] for ns in name_shapes])))
        return

    def restore_checkpoint(self):
        if not hasattr(self, "checkpoint"):
            self.checkpoint_dir = os.path.join(
                self.params.data_path, 'training_checkpoints')
            if not hasattr(self, "optimizer"):
                self.checkpoint = tf.train.Checkpoint(
                    encoder=self.encoder, decoder=self.decoder)
            else:
                self.checkpoint = tf.train.Checkpoint(
                    optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder)
        self.checkpoint.restore(
            tf.train.latest_checkpoint(self.checkpoint_dir))

    def compare_input_output(self, input_array, vocab, max_tar_length, targets=None, beam_search=None):
        """input_array: array representing single sentence"""
        predict_array_input(
            input_array.numpy().tolist(), vocab, max_tar_length,
            self.encoder, self.decoder, targets, beam_search)

    # @tf.function
    def teacher_forcing_test_loss(self, dataval, begin_id):
        """same as train loss calculating method"""
        start = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataval):
            loss = 0
            enc_hidden = self.encoder.initialize_hidden_state(inputs=inp)
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims(
                [begin_id] * inp.shape[0], 1)
            # Teacher forcing - feeding the target as the next input
            for t in tf.range(1, targ.shape[1]):
                # passing enc_output to the decoder
                predictions, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_output)
                loss += self.loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)
            batch_loss = (loss / int(targ.shape[1]))
            # print('Batch {} Loss {:.4f} Time {}'.format(
            #     batch, batch_loss.numpy(), time.time() - start))
            total_loss += batch_loss
        print('Test Loss {:.4f} Time {}'.format(
            total_loss / (batch + 1), time.time() - start))
        return total_loss


class Encoder(tf.keras.Model):
    ''' Encoder(vocab_size, embedding_dim, enc_units, batch_sz)
        encoder input shape (batch_size, max_sequence_length)
        encoder output shape (batch_size, max_length, hidden_size)
        encoder hidden state shape (batch_size, hidden_size)'''

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 enc_units,
                 batch_sz,
                 embedding_matrix=None):
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
        self.lstm = tf.keras.layers.LSTM(enc_units,
                                         return_state=True,
                                         return_sequences=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        state = tf.concat((state_h, state_c), axis=1)
        return output, state

    def initialize_hidden_state(self, inputs=None, batch_size=None):
        if inputs is not None:
            _batch_size = len(inputs)
        elif batch_size is not None:
            _batch_size = batch_size
        else:
            _batch_size = self.batch_size
        return [tf.zeros((_batch_size, self.enc_units)),
                tf.zeros((_batch_size, self.enc_units))]


class Decoder(tf.keras.Model):
    """(self, vocab_size, embedding_dim, dec_units, batch_sz, attention_units,
    attention='BahdanauAttention', embedding_matrix=None)"""

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 dec_units,
                 batch_sz,
                 attention_units,
                 attention='BahdanauAttention',
                 embedding_matrix=None,
                 activation='softmax'):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        print('activation:', activation)
        if embedding_matrix is not None:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim,
                embeddings_initializer=tf.keras.initializers.Constant(
                    embedding_matrix),
                trainable=False)
            self.fc1 = tf.keras.layers.Dense(vocab_size, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(
                embedding_matrix.transpose()), activation=activation, trainable=False)
            print('using pretrained embedding matrix')
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_size, embedding_dim)
            self.fc1 = tf.keras.layers.Dense(
                vocab_size, use_bias=False, activation=activation)
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True)
        self.fc0 = tf.keras.layers.Dense(embedding_dim)
        # used for attention
        if attention == 'BahdanauAttention':
            self.attention = BahdanauAttention(
                attention_units)  # self.dec_units)
            # self.attention = tfaBAttention(attention_units)
        elif attention == 'MyAttention':
            self.attention = MyAttention(attention_units)  # self.dec_units)
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
        # query = tf.concat(hidden, axis=1)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # x shape before: (batch_size, 1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        hidden = hidden[:, :self.dec_units], hidden[:, self.dec_units:]
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab)
        x = self.fc1(self.fc0(output))
        return x, tf.concat((state_h, state_c), axis=1), attention_weights


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


class MyAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        raise NotImplementedError
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
        logging.basicConfig(
            format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        # sentences = MyCorpus()
        modelfasttext = gensim.models.fasttext.FastText(sentences=sentences,
                                                        size=params.embedding_dim, window=5, min_count=params.vocab_min_frequency, workers=3)
        # modelfasttext.save(file)
        embedding_matrix = np.array(
            [modelfasttext.wv[token] for token in sentences.vocab.idx_to_token])
        np.save(file, embedding_matrix, allow_pickle=False)
    return embedding_matrix  # , sentences.vocab


def layer_info(layer):
    name_shapes = [(v.name, v.shape, reduce(lambda x, y: x * y, v.shape))
                   for v in layer.trainable_variables]
    print(
        '\n'.join(['; '.join(
            [nss[0], str(nss[1]), '{:,}'.format(
                reduce(lambda x, y: x * y, nss[1]))]) for nss in name_shapes]))
    print('total trainable weights: {:,}'.format(
        sum([ns[2] for ns in name_shapes])))
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
