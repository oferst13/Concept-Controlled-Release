from datetime import datetime

import best_sol_releases as source
import numpy as np
from matplotlib import pyplot as plt
import benchmark as bm
import pyswmm
from scipy import integrate

swmm_flow = np.zeros((4, bm.sim_len), dtype='float')
tank_flow = np.zeros((3, bm.sim_len), dtype='float')
overflows = source.overflows.copy()
releases = source.releases_volume.copy()
with pyswmm.Simulation('calib.inp') as sim:
    tank1 = pyswmm.Nodes(sim)['Tank1']
    tank2 = pyswmm.Nodes(sim)['Tank2']
    tank3 = pyswmm.Nodes(sim)['Tank3']
    out = pyswmm.Nodes(sim)['outlet']
    j1 = pyswmm.Nodes(sim)['J1']
    j2 = pyswmm.Nodes(sim)['J2']
    j3 = pyswmm.Nodes(sim)['J3']
    sim.start_time = datetime(2021, 1, 1, 0, 0, 0)
    sim.end_time = datetime(2021, 1, 2)
    sim.step_advance(30)
    i = 0
    for step in sim:
        tank_flow[0, i] = 1000*(overflows[0, i] + releases[0, i]) / bm.dt
        tank1.generated_inflow(float(tank_flow[0, i]))
        tank_flow[1, i] = 1000 * (overflows[1, i] + releases[1, i]) / bm.dt
        tank2.generated_inflow(float(tank_flow[1, i]))
        tank_flow[2, i] = 1000 * (overflows[2, i] + releases[2, i]) / bm.dt
        tank3.generated_inflow(float(tank_flow[2, i]))
        swmm_flow[0, i] = j1.total_inflow
        swmm_flow[1, i] = j2.total_inflow
        swmm_flow[2, i] = j3.total_inflow
        swmm_flow[3, i] = out.total_inflow
        i += 1
    print(sim.flow_routing_error)

plt.plot(source.hours[0:bm.zero_Q + 100], source.pipe_Q[2, :bm.zero_Q + 100, 1], label="kinemtic")
plt.plot(source.hours[0:bm.zero_Q + 100], 0.001*swmm_flow[3, :bm.zero_Q + 100], label="dynamic")
plt.legend()
plt.show()
swmm_sum = integrate.simps(swmm_flow*bm.dt/1000, bm.t[0:-1])
print('_')
