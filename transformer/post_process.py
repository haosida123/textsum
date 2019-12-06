import sys
import pandas as pd

if __name__ == "__main__":
    file = sys.argv[1]
    with open(file, 'r') as fp:
        lines = fp.readlines()
    newlines = []
    for line in lines:
        line = ''.join(
            [w for w in line.replace('seperator', '，').replace('\n', '').
                replace('<unk>', '').split(' ') if w not in ['', ' ']])
        if line.endswith('，'):
            line = line[:-1]
        newlines.append(line)
    df = pd.DataFrame(
        zip(["Q{:d}".format(i + 1) for i in range(len(newlines))], newlines))
    df.columns = ['QID', 'Prediction']
    df = df.set_index('QID')
    print(df.head())
    df.to_csv(file[:file.find('.')] + '_result.csv')
#%%
# from preprocessing import gen_cut_csv
# df, df1 = gen_cut_csv()
# df1.head()
# import sys
# import pandas as pd
# file = "predict.txt"
# with open(file, 'r') as fp:
#     lines = fp.readlines()
# newlines = []
# for line in lines:
#     line = ''.join([w for w in line.replace('seperator', '，').replace('\n', '').split(' ') if w not in ['', ' ']])
#     if line.endswith('，'):
#         line = line[:-1]
#     print(line)
#     newlines.append(line)
# df = pd.DataFrame(zip(["Q{:d}".format(i+1) for i in range(len(newlines))], newlines))
# df.columns = ['QID', 'Prediction']
# df.head()

# #%%
# df.index.name = "QID"
# df.to_csv(file[:file.find('.')] + '_result.csv')


# %%
