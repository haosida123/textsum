# from preprocessing import gen_cut_csv
from preprocessing import Vocab, gen_cut_csv, replace_sentence, tokenize_sentence
from utils.config import vocab_train_test_path

from functools import reduce
import numpy as np
import pandas as pd
# import tensorflow as tf


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, min_freq=5, df=None):
        self.min_freq = min_freq
        if not df:
            df, _ = gen_cut_csv('r')
            self.vocab = Vocab.from_json(
                'vocab_dict.json', self.min_freq, use_special_tokens=True)
        else:
            self.vocab = Vocab(df, min_freq=min_freq, use_special_tokens=True)
        self.df = df

    def __iter__(self):
        vocab = self.vocab
        for idx, row in self.df.iterrows():
            # assume there's one document per line, tokens separated by whitespace
            line = row['ColX'].split(' ') + row['Report'].split(' ')
            line = [word for word in line if word]
            yield [vocab.to_tokens(vocab[word]) for word in line]


class TextDataset():
    def __init__(self, df=None, x_cols=None, y_cols=None, min_freq=2, data_size=None):
        """dataframe to text list of lines for textsum
        :param df: pandas.DataFrame
        :param x_cols: predictor columns, cannot be none if df is not None
        :param y_cols: prediction column
        :param datasize: first n lines to test
        train_lines: list with shape (sentences #, sentence length) and 
        corresponding vocabulary for translation"""

        self.min_freq = min_freq

        if df is not None:
            if not isinstance(x_cols, list):
                x_cols = [x_cols]
            if not isinstance(y_cols, list):
                y_cols = [y_cols]
            df.fillna('', inplace=True)
            self.vocab = Vocab(df, min_freq=min_freq, use_special_tokens=True)
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
                self.vocab = Vocab(df, min_freq=min_freq,
                                   use_special_tokens=True)
            else:
                self.vocab = Vocab.from_json(
                    vocab_train_test_path, min_freq=min_freq, use_special_tokens=True)
            self.train_lines_x = self._df2lines(df_train.loc[:, ['ColX']])
            self.train_lines_y = self._df2lines(df_train.loc[:, ['Report']])
            self.test_lines_x = self._df2lines(df_test.loc[:, ['ColX']])

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
        can be used by tf.data.Dataset.from_tensor_slices"""
        x_max_length = self.decide_max_length(
            [len(l) for l in self.train_lines_x])
        y_max_length = self.decide_max_length(
            [len(l) for l in self.train_lines_y])
        npx, self.x_seq_length = self.build_array(
            self.train_lines_x, self.vocab, x_max_length, is_source=True)
        npy, self.y_seq_length = self.build_array(
            self.train_lines_y, self.vocab, y_max_length, is_source=False)
        return npx, npy, self.x_seq_length, self.y_seq_length

    def to_tf_test_input(self, sentences=None):
        """transfer sentences to tokenized numpy array, if no sentences, transfer self.test_lines_x
        can be used by tf.convert_to_tensor or from_tensor_slices"""
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
            [len(l) for l in sentences])
        npx, x_seq_length = self.build_array(
            sentences, self.vocab, x_max_length, is_source=True)
        return npx, x_seq_length

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
    def decide_max_length(lengths):
        """Calculate the 99-percentile of the length of data lines
        lengths: a list of length of entries"""
        lengths.sort()
        ret = np.percentile(lengths, 99)
        print('line min, max, and 99-percentile:\n',
              lengths[0], lengths[-1], ret)
        ret = int(min(np.percentile(lengths, 10) + np.percentile(lengths, 99),lengths[-1]))
        print('selecting:', ret)
        return ret

# datatest = TextDataset()
# tfdx, tfdy, lx, ly = datatest.to_tf_train_input()
# tftest, tftestl = datatest.to_tf_test_input()
# it = iter(tftest)
# ' '.join(datatest.vocab.to_tokens(next(it).tolist()))
