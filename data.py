
from preprocessing import Vocab
import pandas as pd

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __init__(self, min_freq=5):
        self.min_freq = min_freq

    def __iter__(self):
        vocab = Vocab.from_json('vocab_dict.json', self.min_freq)
        df = pd.read_csv('train_cut.csv')
        df.set_index('QID')
        for idx, row in df.iterrows():
            # assume there's one document per line, tokens separated by whitespace
            line =  row['ColX'].split(' ')+row['Report'].split(' ')
            line = [word for word in line if word]
            yield [vocab.to_tokens(vocab[word]) for word in line]