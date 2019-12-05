
import paddle.fluid as fluid

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
        self.lstm = lambda x: fluid.layers.dynamic_lstm(
            x, enc_units * 4, name='encoder_lstm')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        state = tf.concat((state_h, state_c), axis=1)
        return output, state