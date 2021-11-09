import best_sol_releases as source
import numpy as np
from matplotlib import pyplot as plt
import benchmark as bm
import plot_with_rain

plt.figure()
plt.rc('font', size=11)
plot_hours = np.ceil(bm.zero_Q * source.dt / 3600)
t = np.arange(plot_hours+1)
releases_2plot = np.c_[source.release, np.zeros((3, int(plot_hours-source.release.shape[1]+1)))]
ls=['-', '-.', ':']
cl=['xkcd:dark sky blue', 'r', 'xkcd:goldenrod']
for gg, graph in enumerate(releases_2plot):
    plt.step(t, graph*10, cl[gg], where='post', label=f'Tank {gg+1}', linewidth=4-0.7*gg, linestyle=ls[gg])
fig = plt.gcf()
# plt.xticks(np.arange(0, plot_hours+1, 1.0))
fig.set_size_inches(6, 4)
fig.tight_layout(pad=1.5)
plt.legend(loc='upper left')
#, bbox_to_anchor=(0.6,0))
plt.xlabel('t (hours)')
plt.ylabel('Valve Opening %')
plt.xlim([0, plot_hours])
plt.show()