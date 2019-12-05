# from preprocessing import gen_cut_csv
from textsum.preprocessing import Vocab, gen_cut_csv, replace_sentence, tokenize_sentence
from textsum.utils.config import vocab_train_test_path, params, data_path

# import logging
# import gensim
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
# import tensorflow as tf


class TextDataset():
    """text dataset generating numpy arrays from dataframe of text
        Parameters
        ----------
        df: pandas.DataFrame

        x_cols: predictor columns, cannot be none if df is not None

        y_cols: prediction column

        datasize: first n lines to test

        Attributes
        ----------
        train_lines/text_lines: list with shape (sentences #, sentence length) and
        corresponding vocabulary for translation

        vocab: vocabulary which transform between numbers and words
        """

    def __init__(self, df=None, x_cols=None, y_cols=None, data_size=None):
        """from dataframe to text list of lines for textsum"""
        # self.min_freq = params.vocab_min_frequency

        if df is not None:
            if not isinstance(x_cols, list):
                x_cols = [x_cols]
            if not isinstance(y_cols, list):
                y_cols = [y_cols]
            df.fillna('', inplace=True)
            self.vocab = Vocab(df, min_freq=params.vocab_min_frequency,
                               use_special_tokens=True)
            self.train_lines_x = self._df2lines(df[x_cols])
            self.train_lines_y = self._df2lines(df[y_cols])
            self.test_lines_x = None
        else:
            df_train, df_test = gen_cut_csv('r')
            if data_size is not None:
                df_train = df_train.iloc[:data_size]
                df_test = df_test.iloc[:data_size]
                df = pd.concat([df_train, df_test], axis=0,
                               sort=False).fillna('')
                self.vocab = Vocab(df, min_freq=params.vocab_min_frequency,
                                   use_special_tokens=True)
            else:
                self.vocab = Vocab.from_json(
                    vocab_train_test_path, min_freq=params.vocab_min_frequency, use_special_tokens=True)
            try:
                with open(os.path.join(data_path, 'textdatasetxy.json')) as fp:
                    self.train_lines_x, self.train_lines_y, self.test_lines_x = \
                        json.load(fp)
            except Exception as e:
                print('expected', e, '\ncontinuing')
                self.train_lines_x = self._df2lines(df_train.loc[:, ['ColX']])
                self.train_lines_y = self._df2lines(
                    df_train.loc[:, ['Report']])
                self.test_lines_x = self._df2lines(df_test.loc[:, ['ColX']])
                with open(os.path.join(data_path, 'textdatasetxy.json'), 'w') as fp:
                    json.dump(
                        (self.train_lines_x, self.train_lines_y, self.test_lines_x), fp)

    def _df2lines(self, df):
        print('tokenizing lines')

        def combine(row):
            line = [row[column].split(' ') for column in df.columns]
            line = reduce(lambda x, y: x + y, line)  # combine all columns
            line = [word for word in line if word not in ['', ' ']]
            return [self.vocab[word] for word in line]
        lines = df.apply(combine, axis=1).to_list()
        return lines

    def to_tf_train_input(self):
        """transfer x_seq_length, y_seq_lengthto tokenized numpy array,
        can be used by tf.data.Dataset.from_tensor_slices
        Returns
        -------
        npx: numpy array
            test input sentences
        x_sequence_length: numpy array
            corresponding length of input sentences
        npy: numpy array
            test output sentences
        y_sequence_length: numpy array
            corresponding length of output sentences
        """
        x_max_length = self.decide_max_length(
            [len(l) for l in self.train_lines_x],
            params.decide_length_percentile1,
            params.decide_length_percentile2
        )
        y_max_length = self.decide_max_length(
            [len(l) for l in self.train_lines_y],
            params.decide_length_percentile1,
            params.decide_length_percentile2
        )
        npx, self.x_seq_length = self.build_array(
            self.train_lines_x, self.vocab, x_max_length, is_source=True)
        npy, self.y_seq_length = self.build_array(
            self.train_lines_y, self.vocab, y_max_length, is_source=False)
        return npx, npy, self.x_seq_length, self.y_seq_length

    def to_tf_test_input(self, sentences=None):
        """transfer sentences to tokenized numpy array, if no sentences, transfer self.test_lines_x
        can be used by tf.convert_to_tensor or from_tensor_slices
        Returns
        -------
        npx: numpy array
            test input sentences
        x_sequence_length: numpy array
            corresponding length of input sentences
        """
        if sentences:
            if isinstance(sentences, str):
                sentences = [sentences]
            sentences = [[self.vocab[word] for word in tokenize_sentence(
                replace_sentence(s))] for s in sentences]
        elif self.test_lines_x is not None:
            sentences = self.test_lines_x
        else:
            raise (RuntimeError, 'no sentences input or test dataframe')
        x_max_length = self.decide_max_length(
            [len(l) for l in sentences],
            params.decide_length_percentile1,
            params.decide_length_percentile2
        )
        npx, x_seq_length = self.build_array(
            sentences, self.vocab, x_max_length, is_source=True)
        return npx, x_seq_length

    def to_generator(self, x, y):
        """return a generator for data reading and bucketing"""
        y = [[self.vocab.bos] + l + [self.vocab.eos] for
             l in y]
        return lambda: zip(x, y)
        # used for: bucket_by_sequence_length
        # tf.data.experimental.bucket_by_sequence_length
        # https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length
        # https://stackoverflow.com/questions/50606178/tensorflow-tf-data-dataset-and-bucketing

    def trim_train_lines(self):
        print("x length:")
        x_max_length = self.decide_max_length(
            sorted([len(x) for x in self.train_lines_x]),
            params.bucket_length_percentile1,
            params.bucket_length_percentile2)
        self.train_lines_x = [line[:x_max_length] for line in
                              self.train_lines_x]
        print("y length:")
        y_max_length = self.decide_max_length(
            sorted([len(y) for y in self.train_lines_y]),
            params.bucket_length_percentile1,
            params.bucket_length_percentile2)
        self.train_lines_y = [line[:y_max_length] for line in
                              self.train_lines_y]
        return x_max_length, y_max_length

    @staticmethod
    def get_bucketing_boundaries_batchsizes(lengths, batchsize):
        lengths.sort()
        steps, last_batch = divmod(
            len(lengths), batchsize)  # total # of batches
        batchsizes = [batchsize] * steps + [last_batch]
        bounds = np.percentile(lengths, np.arange(0, 100, 100/steps)[:steps])
        bounds = [int(b) for b in bounds]
        return bounds, batchsizes, steps

    @staticmethod
    def trim_pad(line, num_steps, padding_token):
        if len(line) > num_steps:
            return line[:num_steps]  # Trim
        return line + [padding_token] * (num_steps - len(line))  # Pad

    @staticmethod
    def build_array(lines, vocab, num_steps, is_source):
        # lines = [vocab[l] for l in lines] # already transorformed
        if not is_source:
            lines = [[vocab.bos] + l + [vocab.eos] for l in lines]
        array = np.array([TextDataset.trim_pad(
            l, num_steps, vocab.pad) for l in lines])
        valid_len = (array != vocab.pad).sum(axis=1)
        return array, valid_len

    @staticmethod
    def decide_max_length(lengths, percentile1, percentile2):
        """Calculate the 99-percentile of the length of data lines
        lengths: a list of length of entries"""
        lengths.sort()
        ret = np.percentile(lengths, 99)
        print('line percentiles:\n',
              np.percentile(lengths, list(range(0, 101, 10))))
        ret = int(min(
            np.percentile(lengths, percentile1) +
            np.percentile(lengths, percentile2),
            lengths[-1]))
        # ret = lengths[-1]
        print('selecting:', ret)
        return ret

    def element_length(self, array):
        """return length of array without padding"""
        return sum(array != self.vocab.pad)


def transformer_reader(mode='train'):
    """mode = train/val/test"""
    textdata = TextDataset()

    def bos_eos(lines):
        return ([[textdata.vocab.bos] + line + [textdata.vocab.eos] for
                line in lines])
    if mode == 'test':
        x_max_length = TextDataset.decide_max_length(
            sorted([len(x) for x in textdata.train_lines_x]),
            params.bucket_length_percentile1,
            params.bucket_length_percentile2)
        textdata.test_lines_x =\
            bos_eos([line[:x_max_length] for line in textdata.test_lines_x])
        word_dict = {}
        for i, token in enumerate(textdata.vocab.idx_to_token):
            word_dict[i] = token
        return lambda: iter(textdata.test_lines_x), word_dict
    else:
        x_max_length, y_max_length = textdata.trim_train_lines()
        x_train, x_val, y_train, y_val = train_test_split(
            textdata.train_lines_x, textdata.train_lines_y,
            test_size=0.01, random_state=53)
        if mode == 'train':
            x, y0 = bos_eos(x_train), y_train
        elif mode == 'val':
            x, y0 = bos_eos(x_val), y_val
        else:
            raise ValueError
        y = [[textdata.vocab.bos] + line for line in y0]
        y_next = [line + [textdata.vocab.eos] for line in y0]

        def generator():
            for xyy in zip(x, y, y_next):
                yield xyy
        return generator


def process_sentence_to_feed(sentence, vocab, max_length_inp):
    sentence = tokenize_sentence(replace_sentence(sentence))
    inputs = [vocab[w] for w in sentence]
    # inputs = tf.keras.preprocessing.sequence.pad_sequences(
    #     [inputs], maxlen=max_length_inp, padding='post')
    inputs = TextDataset.build_array(
        [inputs], vocab, max_length_inp, is_source=True)[0]
    # result = evaluate(
    #     inputs, vocab, max_length_targ, encoder, decoder)
    return inputs
