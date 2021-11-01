import best_sol_releases as source
import numpy as np
from matplotlib import pyplot as plt
import benchmark as bm

plt.rc('font', size=14)
plot_hours = np.ceil(bm.zero_Q * source.dt / 3600)
t = np.arange(np.shape(source.release)[1] + 1)
releases_2plot = np.c_[source.release, np.zeros(3)]
ls=['-','-.',':']
cl=['xkcd:dark sky blue','r','xkcd:goldenrod']
for gg, graph in enumerate(releases_2plot):
    plt.step(t, graph*10, cl[gg], where='post', label=f'Tank {gg+1}', linewidth=4-0.7*gg, linestyle=ls[gg])
plt.legend()
plt.show()