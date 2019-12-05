#%%
import collections
import logging
import sys
try:
    import gensim
    import jieba
except ImportError as e:
    print(e)
    if input('continue?(y/n)').lower() == 'n':
        sys.exit(1)
import pandas as pd
import os
import re
import numpy as np
import json
import sys
sys.path.append("..")
from textsum.utils.config import train_data_path, test_data_path
from textsum.utils.config import train_seg_path, test_seg_path
from textsum.utils.config import user_dict, vocab_train_test_path  # , vocab_train_path
from textsum.utils.config import params

URLPATTERN = re.compile(
    "(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9\@\:\%\_\+\~\#\=]{1,256}\.[-a-z]{1,6}[\.\\\/]{1,2}[-\w]{1,6}([-a-zA-Z0-9@:%_\+.~#?&//\=]*)")
TBPATTERN = "【.+点击链接，再选择浏览器打开.+打开\?\?手淘\?\?([来自超级会员的分享])?"
REPLIST = [('\[图片\]', ' imagemsg '), ('\[语音\]', ' voicemsg '),
            ('\[视频\]', ' videomsg '), ('㎜', ' mm '), ('㎞', ' km '),
            ('㎏', ' kg '), ('℃', ' 摄氏度 '), ('℉', ' 华氏度 '),
            ("\$", " 美元 "), ("￥", " 人民币 "), ("\%", " 百分之 "),
            (URLPATTERN, " netpage "), (TBPATTERN, " tbshare "),
            ("(?<=\d)，(?=000)", "")]


def read_auto_master(test=False):
    """return trainset, testset"""
    df = pd.read_csv(train_data_path)
    df.set_index('QID', inplace=True)
    colx = ['Question', 'Dialogue']
    coly = ['Report']
    # df = df.loc[:,colx+coly].dropna()
    df = df.drop(
        index=df.index[df.loc[:, colx+coly].isnull().apply(any, axis=1)])
    print('read train set.')
    if not test:
        return(df)
    df1 = pd.read_csv(test_data_path)
    df1.set_index('QID', inplace=True)
    print('read test set.')
    return(df, df1)


def replace_sentence(sentence):
    for fro, to in REPLIST:
        sentence = re.sub(fro, to, str(sentence))
    return sentence


def replace_media_msg(df, dropindex=['Q14105', 'Q2085', 'Q46993', 'Q62199', 'Q73829', 'Q74003', 'Q26822', 'Q28927', 'Q72231', 'Q76179']):
    """clean & replace: [('[图片]', 'imagemsg'), ('[语音]', 'voicemsg'), ('[视频]', 'videomsg')]"""
    # Q26822', 'Q28927', 'Q72231', 'Q76179': No valid quesion
    print('replacing.')
    # **** Clean ****   # train set
    if 'Report' in df.columns:  # train set
        try:
            a = "data:image/png;base64，`+MDEyMDg0MjU1Nzg0MDI4NjAyNDMy`+"
            df.loc['Q69653', 'Dialogue'] = df.loc['Q69653',
                                                  'Dialogue'].replace(a, "")
            df.Report['Q26326'] = "嘉实多"
            df.drop(index=dropindex, inplace=True)
        except Exception:
            pass

    # **** Replace ****
    for col in df.columns:
        df[col] = df[col].apply(replace_sentence).astype('str')
        # for fro, to in replist:
        #     df[col] = df[col].apply(lambda s: re.sub(fro, to, str(s))).astype('str')
    with open(user_dict, 'w', encoding='utf-8') as file:
        file.write('seperator'+'\n')
        for rep in REPLIST:
            file.write(rep[1].strip()+'\n')


def tokenize_sentence(sentence, sep=True):
    # sentence = replace_sentence(sentence)
    jieba.load_userdict(user_dict)
    patternnonword = re.compile("[^\w ]+")
    if sep:
        patternsep = re.compile("[,|，|.|。|；|;|:|：|!|！|?|？]+")
        sentence = re.sub(patternsep, " seperator ", str(sentence))
    sentence = re.sub(patternnonword, "", str(sentence))
    sentence = [word for word in jieba.cut(sentence) if (word not in ['', ' '])]
    return sentence


def tokenize(df, sep=True):
    """Split sentences. Col_Report w/wo comma"""
    jieba.load_userdict(user_dict)
    print('tokenizing.')
    # patternnonword = re.compile("[^\w ]+")
    # patternsep = re.compile("[,|，|.|。|；|;|:|：|!|！|?|？]+")
    for col in df.columns:
        # if sep:
        #     df[col] = df[col].apply(lambda s: re.sub(
        #         patternsep, " seperator ", str(s))).astype('str')
        # df[col] = df[col].apply(lambda s: re.sub(
        #     patternnonword, "", str(s))).astype('str')
        # df[col] = df[col].apply(lambda s: ' '.join(jieba.cut(s)))
        df[col] = df[col].apply(lambda s: ' '.join(tokenize_sentence(s, sep)))
    colx = ['Brand', 'Model', 'Question', 'Dialogue']
    df['ColX'] = df.apply(lambda x: ' seperator '.join([x[col] for col in colx]), axis=1)
    df.drop(columns=colx, inplace=True)


def gen_cut_csv(mode='r'):
    print('get cut text.')
    if mode == 'r' or mode == 'read':
        try:
            print('read from file.')
            dfcut = pd.read_csv(train_seg_path)
            df1cut = pd.read_csv(test_seg_path)
            dfcut.set_index('QID', inplace=True)
            df1cut.set_index('QID', inplace=True)
            dfcut.fillna("", inplace=True)
            df1cut.fillna("", inplace=True)
            return(dfcut, df1cut)
        except FileNotFoundError as e:
            print(e, '\ncontinue.')
            mode = 'w'
    if mode == 'w' or mode == 'write':
        print('generating cut from raw.')
        dfcut, df1cut = read_auto_master(True)
        replace_media_msg(dfcut)
        replace_media_msg(df1cut)
        tokenize(dfcut, True)
        dfcut.fillna("", inplace=True)
        dfcut.to_csv(train_seg_path)
        tokenize(df1cut, True)
        df1cut.fillna("", inplace=True)
        df1cut.to_csv(test_seg_path)
        return(dfcut, df1cut)
    else:
        raise(TypeError, 'mode=w/r')


def count_corpus(sentences):
    """count sentences/dataframe tokens"""
    if isinstance(sentences, list):
        # Flatten a list of token lists into a list of tokens
        tokens = [tk for line in sentences for tk in line]
    elif isinstance(sentences, pd.DataFrame):
        sentences.fillna("", inplace=True)
        tklists = [data.split(
            ' ') for column in sentences.columns for data in sentences[column].to_list()]
        tokens = [tk for tklist in tklists for tk in tklist if (tk not in [
                                                                '', ' '])]
        # print(tokens[:500])
    else:
        raise(TypeError)
    return(collections.Counter(tokens))


class Vocab(object):
    def __init__(self, tokens, min_freq=0, use_special_tokens=False):
        # Sort according to frequencies from corpus/dict
        print('generating vocab.')
        if not isinstance(tokens, dict):
            counter = count_corpus(tokens)
        else:
            counter = tokens
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        if use_special_tokens:
            # padding, begin of sentence, end of sentence, unknown
            self.pad, self.bos, self.eos, self.unk = (0, 1, 2, 3)
            uniq_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        else:
            self.unk, uniq_tokens = 0, ['<unk>']
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if isinstance(indices, (list, tuple)):
            return [self.to_tokens(index) for index in indices]
        return self.idx_to_token[indices]

    def to_json(self, path):
        with open(path, 'w') as file:
            json.dump(dict(self.token_freqs), file)

    @staticmethod
    def from_json(path, min_freq=0, use_special_tokens=False):
        with open(path, 'r') as file:
            counter = json.load(file)
        return Vocab(counter, min_freq, use_special_tokens)


class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, df=None):
        # self.min_freq = params.vocab_min_frequency
        if df is None:
            dfcut, df1cut = gen_cut_csv('r')
            df = pd.concat([dfcut, df1cut], axis=0, sort=False
                           ).fillna('')
            self.vocab = Vocab.from_json(
                vocab_train_test_path, min_freq=params.vocab_min_frequency, use_special_tokens=True)
        else:
            self.vocab = Vocab(df, min_freq=params.vocab_min_frequency,
                               use_special_tokens=True)
        self.df = df

    def __iter__(self):
        vocab = self.vocab
        for idx, row in self.df.iterrows():
            # assume there's one document per line, tokens separated by whitespace
            line = row['ColX'].split(' ') + row['Report'].split(' ')
            line = [word for word in line if word]
            yield vocab.to_tokens([vocab.bos]) + [vocab.to_tokens(vocab[word]) for word in line] + vocab.to_tokens([vocab.eos] + [vocab.pad] * 20)


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
            raise RuntimeError('arg sentences should be a generator of sentences')
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


if __name__ == '__main__':
    dfcut, df1cut = gen_cut_csv('r')
    # dfcut, df1cut = gen_cut_csv('write')
    # vocab = Vocab(dfcut, min_freq=0)
    # print('vocab len, vocab total:')
    # total = sum([x[1] for x in vocab.token_freqs])
    # print(len(vocab.token_freqs), total)
    # plt.loglog(list(range(len(vocab.token_freqs))), [x[1]/total for x in vocab.token_freqs])
    # plt.show()
    # vocab.to_json(vocab_train_path)
    # vocab_from_json = Vocab.from_json(vocab_train_path)

    df = pd.concat([dfcut, df1cut], axis=0, sort=False
                   ).fillna('')
    vocab = Vocab(df, min_freq=0, use_special_tokens=True)
    vocab.to_json(vocab_train_test_path)
    print('vocab len, vocab total:')
    total = sum([x[1] for x in vocab.token_freqs])
    print(len(vocab.token_freqs), total)
    # vocab_from_json = Vocab.from_json(vocab_train_test_path)
    # print(sorted(list(vocab.token_freqs.keys())) == sorted(list(vocab_from_json.token_freqs.keys())))
    fasttext_embedding(params, sentences=MyCorpus())
#pd.set_option('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', 500)


# %%
