# %%
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import tensorflow as tf
from data import TextDataset  # , MyCorpus
from utils.config import params
from tf_seq2seq_att import fasttext_embedding, Seq2seq_attention
from beam_search import BeamSearch
# from rouge_l_tensorflow import tf_rouge_l


def load_data_trim(params):
    print("load_data_trim")
    datatext = TextDataset()
    tfdx, tfdy, seq_length_x, seq_length_y = datatext.to_tf_train_input()
    # tftest, tftestl = datatext.to_tf_test_input()
    # it = iter(tfdx)
    # print(' '.join(vocab.to_tokens(next(it).tolist())))
    input_tensor_train, input_tensor_val, \
        target_tensor_train, target_tensor_val = train_test_split(
            tfdx, tfdy, test_size=0.01, random_state=53)
    print(len(input_tensor_train), len(target_tensor_train),
          len(input_tensor_val), len(target_tensor_val))
    BUFFER_SIZE = len(input_tensor_train)
    steps_per_epoch = len(input_tensor_train)//params.batch_size

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(params.batch_size, drop_remainder=True)
    dataval = tf.data.Dataset.from_tensor_slices(
        (input_tensor_val, target_tensor_val)).batch(
            params.batch_size)  # , drop_remainder=True)
    return dataset, dataval.unbatch(), max(seq_length_y), steps_per_epoch, datatext.vocab


def load_data_buckets(params):
    print("load_data_buckets")
    textdata = TextDataset()
    x_train, x_val, y_train, y_val = train_test_split(
        textdata.train_lines_x, textdata.train_lines_y,
        test_size=0.01, random_state=53)
    print("data size:")
    print(len(x_train), len(x_val), len(y_train), len(y_val))
    dataset = tf.data.Dataset.from_generator(
        textdata.to_generator(x_train, y_train),
        output_types=(tf.int32, tf.int32),
        output_shapes=([None], [None])).shuffle(len(x_train))
    bucket_boundaries, batchsizes, steps_per_epoch = \
        TextDataset.get_bucketing_boundaries_batchsizes(
            [len(l) for l in x_train], params.batch_size)
    print("boundaries:", [bucket_boundaries[i] for i in range(0,
        steps_per_epoch, steps_per_epoch//10) if i < len(bucket_boundaries)])
    print("step size", steps_per_epoch)
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
        lambda x, y: tf.shape(x)[0],
        bucket_boundaries,
        batchsizes,
        padding_values=(textdata.vocab.pad, textdata.vocab.pad),
        drop_remainder=True))
        # padded_shapes=([None], [None])))

    dataval = tf.data.Dataset.from_generator(
        textdata.to_generator(x_val, y_val),
        output_types=(tf.int32, tf.int32),
        output_shapes=([None], [None])).shuffle(len(x_val))
    y_max_length = max([len(l) for l in textdata.train_lines_y])
    return dataset, dataval, y_max_length, steps_per_epoch, textdata.vocab


def load_test_data(params):
    print("load_test_data")


def train(dataset, dataval, y_max_length, steps_per_epoch, vocab, epochs, restore, params):
    print("training")
    print(params.__dict__)
    # seq2seq = Seq2seq_attention(len(vocab), params)
    seq2seq = Seq2seq_attention(
        len(vocab), params, embedding_matrix=fasttext_embedding(params, sentences=None))
    seq2seq.encoder.embedding.trainable = False
    seq2seq.decoder.embedding.trainable = False
    seq2seq.decoder.fc1.trainable = False
    # it = iter(dataval)
    # inp, out = next(it)
    beam_search = BeamSearch(seq2seq, params.beam_size, vocab.bos,
                             vocab.eos, y_max_length)
    # seq2seq.compare_input_output(
    #     inp, vocab, y_max_length, out, beam_search)
    seq2seq.summary()
    # seq2seq.encoder.summary()
    # seq2seq.decoder.summary()

    # def my_loss(truth, preds):
    #     return sum(tf_rouge_l(preds, truth, vocab.eos))
    def callback():
        it = iter(dataval)
        for _ in range(3):
            inp, out = next(it)
            seq2seq.compare_input_output(
                inp, vocab, y_max_length, out, beam_search)

    # seq2seq.train_epoch(dataset, epochs, steps_per_epoch, vocab.bos,
    #                     restore_checkpoint=True, dataval=dataval, callback=None)
    seq2seq.train_epoch(
        dataset, epochs, steps_per_epoch, vocab.bos,
        restore_checkpoint=restore, dataval=dataval,
        callback=callback)


def check_train_results(dataval, y_max_length, steps_per_epoch, vocab, params):
    print("check_train_results")
    seq2seq = Seq2seq_attention(
        len(vocab), params, embedding_matrix=fasttext_embedding(params, sentences=None))
    seq2seq.encoder.embedding.trainable = False
    seq2seq.decoder.embedding.trainable = False
    seq2seq.decoder.fc1.trainable = False
    it = iter(dataval)
    inp, out = next(it)
    beam_search = BeamSearch(seq2seq, 9, vocab.bos,
                             vocab.eos, y_max_length)
    # seq2seq.compare_input_output(inp, vocab, y_max_length, out, beam_search)
    seq2seq.summary()

    def callback():
        for _ in range(10):
            inp, out = next(it)
            seq2seq.compare_input_output(
                inp, vocab, y_max_length, out, beam_search)

    # seq2seq.train_epoch(dataset, 5, steps_per_epoch, vocab.bos,
    #                     restore_checkpoint=True, dataval=dataval, callback=None)
    # seq2seq.train_epoch(dataset, 5, steps_per_epoch, vocab.bos,
    #                     restore_checkpoint=True, dataval=dataval, callback=callback)
    seq2seq.restore_checkpoint()
    callback()


# %%
if __name__ == '__main__':
    parser = ArgumentParser(description='train model from data')
    parser.add_argument("-t", "--train_epochs", type=int, default=0
                        )
    parser.add_argument("-b", "--use_bucket", default="false", type=str
                        )
    parser.add_argument("-r", "--restore_checkpoint", default="True", type=str
                        )
    parser.add_argument("-s", "--batch_size", type=int, default=4
                        )
    print(params.__dict__)
    args = parser.parse_args()
    params.batch_size = args.batch_size
    print(args.__dict__)
    if args.use_bucket.lower() == 'true':
        dataset, dataval, y_max_length, steps_per_epoch, vocab = \
            load_data_buckets(params)
    elif args.use_bucket.lower() == 'false':
        dataset, dataval, y_max_length, steps_per_epoch, vocab = \
            load_data_trim(params)
    else:
        raise TypeError
    if args.train_epochs > 0:
        if args.restore_checkpoint.lower() == 'true':
            restore = True
        elif args.restore_checkpoint.lower() == 'false':
            restore = False
        else:
            raise TypeError
        train(dataset, dataval, y_max_length, steps_per_epoch,
              vocab, args.train_epochs, restore, params)
    # elif args.mode == 'test':
    #     test(args)

    # tf.config.experimental_run_functions_eagerly(True)

    check_train_results(dataval, y_max_length, steps_per_epoch, vocab, params)


#%%
params.batch_size=2
dataset, dataval, y_max_length, steps_per_epoch, vocab = \
            load_data_trim(params)
seq2seq = Seq2seq_attention(
    len(vocab), params, embedding_matrix=fasttext_embedding(params, sentences=None))
seq2seq.encoder.embedding.trainable = False
seq2seq.decoder.embedding.trainable = False
seq2seq.decoder.fc1.trainable = False
it = iter(dataval)
inp, out = next(it)
beam_search = BeamSearch(seq2seq, 9, vocab.bos,
                            vocab.eos, y_max_length)
# seq2seq.compare_input_output(inp, vocab, y_max_length, out, beam_search)
seq2seq.summary()

def callback():
    for _ in range(10):
        inp, out = next(it)
        seq2seq.compare_input_output(
            inp, vocab, y_max_length, out, beam_search)

# seq2seq.train_epoch(dataset, 5, steps_per_epoch, vocab.bos,
#                     restore_checkpoint=True, dataval=dataval, callback=None)
# seq2seq.train_epoch(dataset, 5, steps_per_epoch, vocab.bos,
#                     restore_checkpoint=True, dataval=dataval, callback=callback)
seq2seq.restore_checkpoint()
# callback()

# %%
seq2seq.teacher_forcing_test_loss(dataval, vocab.bos)

# %%
