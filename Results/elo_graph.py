import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

PATH = '../Saves/Run01/'

ELO1 = np.load(PATH + f'ELO1.npy')
ELO2 = np.load(PATH + f'ELO2.npy')

plt.plot(ELO1, label='ELO1')
plt.plot(ELO2, label='ELO2')

plt.legend()

plt.show()
