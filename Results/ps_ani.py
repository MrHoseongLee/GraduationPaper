import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from itertools import count

pss = []

PATH = '../Saves/Run03/'

files = os.listdir(PATH + 'PS/')
files.sort()

for file in files:
    pss.append(np.load(PATH + f'PS/{file}'))

index = count(0)
next(index)

def update(_):
    i = next(index)
    ps = pss[i]

    if i > 6250: ani.event_source.stop()

    plt.cla()
    plt.ylim(0, 0.02)
    plt.plot(ps)

ani = animation.FuncAnimation(plt.gcf(), update, interval=10)

plt.tight_layout()
plt.show()
