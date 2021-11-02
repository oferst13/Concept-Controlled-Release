import best_sol_releases as source
import numpy as np
from matplotlib import pyplot as plt
import benchmark as bm

plt.rc('font', size=11)
plot_hours = np.ceil(bm.zero_Q * source.dt / 3600)
fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
fig.set_size_inches(6,4)
axs[0].bar(source.rain_hours[np.nonzero(source.rain_hours <= plot_hours)],
           source.rain[0:len(source.rain_hours[np.nonzero(source.rain_hours <= plot_hours)])], width=source.rain_dt / 3600,
           align='edge')
axs[0].spines['bottom'].set_visible(False)
# axs[0].axes.xaxis.set_visible(False)
axs[0].tick_params(labelbottom=False)
axs[0].set_xlim([0, plot_hours])
# axs[0].set_ylim([0,5])
axs[0].set_ylabel('Rain\n (mm/10-minutes)')
axs[0].invert_yaxis()
axs[0].grid(True)
axs[1].plot(source.hours[np.nonzero(source.hours <= plot_hours)],
            1000*bm.pipe_Q[2, 0:len(source.hours[np.nonzero(source.hours <= plot_hours)]), 1], 'r-',
            label="Benchmark")
axs[1].plot(source.hours[np.nonzero(source.hours <= plot_hours)],
            1000*source.pipe_Q[2, 0:len(source.hours[np.nonzero(source.hours <= plot_hours)]), 1], 'b-',
            label="Controlled")
axs[1].plot(source.hours[np.nonzero(source.hours <= 1+bm.last_overflow*bm.dt/3600)],
            1000*np.ones(len(source.hours[np.nonzero(source.hours <= 1+bm.last_overflow*bm.dt/3600)])) * bm.obj_Q,
            'g--', label="$Q_{objective}$")
axs[1].set_ylabel('Outfall Flow Rate (LPS)')
axs[1].set_xlabel('t (hours)')
axs[1].set_xlim([0, plot_hours])
axs[1].set_ylim(bottom=0)
axs[1].spines['top'].set_visible(False)
axs[1].grid(True)
fig.tight_layout(pad=0)
plt.legend()
plt.show()