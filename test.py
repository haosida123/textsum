#%%
from load_data import TextDataset
td = TextDataset()

# %%
import numpy as np
# td = TextDataset()
lenx = np.array([len(line) for line in td.train_lines_x])
leny = np.array([len(line) for line in td.train_lines_y])
remove = (lenx < leny - 2) & (lenx > leny -5)
npliney = np.array(td.train_lines_y)
nplinex = np.array(td.train_lines_x)
[list(map(lambda x: ''.join(td.vocab.to_tokens(x)).replace('seperator', ','), [i,j]))
 for (i, j) in zip(nplinex[remove], npliney[remove])]

#%%
td = TextDataset()
lenx = np.array([len(line) for line in td.train_lines_x])
leny = np.array([len(line) for line in td.train_lines_y])
keep = lenx >= leny - 5
td.train_lines_x = np.array(td.train_lines_x)[keep]
td.train_lines_y = np.array(td.train_lines_y)[keep]
print(sum(keep), len(keep))
td.train_lines_y = [[t for t in line if t != td.vocab.unk]
                    for line in td.train_lines_y]
count = [len([t for t in line if td.vocab.to_tokens(t) != 'seperator'])
            for line in td.train_lines_y]
keep = np.array([c >= 2 for c in count])
npliney = np.array(td.train_lines_y)
td.train_lines_x = np.array(td.train_lines_x)[keep]
td.train_lines_y = np.array(td.train_lines_y)[keep]
print(sum(keep), len(keep))


#%%

from preprocessing import gen_cut_csv
df, df1 = gen_cut_csv()

# %%
import pandas as pd
df.head()
pd.set_option('display.max_rows', 5,
    'display.max_columns', None, 'display.max_colwidth', 5000)
# dfres = df.loc[df.Report.apply(lambda x: len(x.split(' ')) < 2)]
dfres = df.loc[df.Report.apply(lambda x: 'nan' in (x.split(' ')))]
dfres = dfres.applymap(lambda x: ''.join(
    x.replace('seperator', ',').split(' ')))
dfres
