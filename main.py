import numpy
import numpy as np
from collections import namedtuple


def tank_fill(tank_storage, rain, tank_size):
    overflows = np.zeros_like(tank_size, dtype=float)
    for tank_num, tank in enumerate(tank_size):
        tank_storage[tank_num] = tank_storage[tank_num] + rain[tank_num]
        if tank_storage[tank_num] > tank:
            overflows[tank_num] = tank_storage[tank_num] - tank
            tank_storage[tank_num] = tank
        res = namedtuple("tank_overflows", ["tank_storage", "overflows"])
    return res(tank_storage, overflows)


dt = 30
rain_dt = 600
beta = 5 / 4
manning = 0.012
sim_len = 600
t = np.linspace(0, sim_len, num=sim_len + 1)
t = t.astype(int)

tank_size = np.array([2, 2, 2])
tank_storage = np.zeros_like(tank_size, dtype=np.longfloat)
roof = np.array([1000, 1000, 1000])

tank_outlets = np.array([500, 500, 500])
tank_Ds = np.array([0.2, 0.2, 0.2])
tank_slopes = np.array([0.02, 0.02, 0.02])
tank_alphas = (0.501 / manning) * (tank_Ds ** (1 / 6)) * (tank_slopes ** 0.5)
c_tanks = tank_outlets / dt

pipes_L = np.array([1000, 1000, 1000])
pipe_Ds = np.array([0.5, 0.5, 0.5])
pipe_slopes = np.array([0.02, 0.02, 0.02])
pipe_alphas = (0.501 / manning) * (pipe_Ds ** (1 / 6)) * (pipe_slopes ** 0.5)
c_pipes = pipes_L / dt

rain_10min = np.linspace(0, 0.6, 4)
rain_10min = np.append(rain_10min, np.flip(rain_10min))
rain_size = rain_dt / dt
rain = np.array([])
for rain_I in rain_10min:
    rain = np.append(rain, np.ones(int(rain_size)) * rain_I)
rain = np.append(rain, np.zeros(sim_len - len(rain)))
rain_volume = np.matmul(np.reshape(roof, (len(roof), 1)), np.reshape(rain, (1, len(rain)))) / 1000
overflows = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)

outlet_A = np.zeros((len(tank_outlets), sim_len, 2), dtype=np.longfloat)
outlet_Q = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
pipe_A = np.zeros((len(pipes_L), sim_len, 2), dtype=np.longfloat)
pipe_Q = pipe_A

warning = 0

for i in range(1, sim_len):
    fill_result = tank_fill(tank_storage, rain_volume[:, i], tank_size)
    tank_storage = fill_result.tank_storage
    overflows[:, i] = fill_result.overflows
    outlet_A[:, i, 0] = ((overflows[:, i] / dt) / tank_alphas) ** (1/beta)
    for j in range(len(tank_outlets)):
        outlet_A[j, i, 1] = outlet_A[j, i-1, 1] - tank_alphas[j] * beta * (dt / tank_outlets[j]) * (((outlet_A[j, i - 1, 0] + outlet_A[j, i-1, 1]) / 2.0) ** (beta - 1)) * (outlet_A[j, i-1, 1] - outlet_A[j, i - 1, 0])
    outlet_Q[:, i] = tank_alphas * (outlet_A[:, i, 1] ** beta)
    for j in range(len(pipes_L)):
        if j > 0:
            pipe_Q[j, i, 0] = pipe_Q[j-1, i, 1] + outlet_Q[j, i]
        else:
            pipe_Q[j, i, 0] = outlet_Q[j, i]
        pipe_A[j, i, 0] = (pipe_Q[j, i, 0] / pipe_alphas[j]) ** (1/beta)
        constants = pipe_alphas[j] * beta * (dt / pipes_L[j])
        pipe_A[j, i, 1] = pipe_A[j, i-1, 1] - constants * (((pipe_A[j, i-1, 0] + pipe_A[j, i-1, 1]) / 2) ** (beta - 1)) * (pipe_A[j, i-1, 1] - pipe_A[j, i-1, 0])
        print(str(i) + ' ' + str(j) + ' ' + str(pipe_A[j, i, 1]))
        pipe_Q[j, i, 1] = pipe_alphas[j] * (pipe_A[j, i, 1] ** beta)
        b=1

print(rain_volume)

