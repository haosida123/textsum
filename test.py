#%%
# import numpy as np
# import matplotlib.pyplot as plt
# a = np.arange(1, 75)
# plt.plot(a, 1/np.exp(a/100))


# %%
import time
import sys
for idx in range(10):
    print("\r{0}".format(idx), end='')
    time.sleep(0.2)

#%%

print("{:.0f}".format(2.235235233))