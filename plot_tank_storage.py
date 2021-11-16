import best_sol_releases as source
import numpy as np
from matplotlib import pyplot as plt
import benchmark as bm
import plot_with_rain
import plot_release_policy


plt.figure()
plt.rc('font', size=11)
plot_hours = np.ceil(bm.zero_Q * source.dt / 3600)
t = np.arange(plot_hours+1)
plot_storage = source.tank_storage_all
ls=['-', '-.', ':']
cl=['xkcd:dark sky blue', 'r', 'xkcd:goldenrod']
for gg, graph in enumerate(plot_storage):
    plt.plot(source.hours[np.nonzero(source.hours <= plot_hours)], graph[np.nonzero(source.hours <= plot_hours)], cl[gg]
             , label=f'Tank {gg+1}', linewidth=4-0.7*gg, linestyle=ls[gg])
fig = plt.gcf()
# plt.xticks(np.arange(0, plot_hours+1, 1.0))
fig.set_size_inches(6, 3.75)
fig.tight_layout(pad=1.5)
plt.legend(loc='center right')
#, bbox_to_anchor=(0.6,0))
plt.xlabel('t (hours)')
plt.ylabel('Tank storage ($m^3$)')
plt.xlim([0, plot_hours])
plt.show()