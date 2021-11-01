import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from timer import Timer
from scipy import integrate


def tank_fill(tank_storage, rain, tank_size):
    overflows = np.zeros_like(tank_size, dtype=float)
    for tank_num, tank in enumerate(tank_size):
        tank_storage[tank_num] = tank_storage[tank_num] + rain[tank_num]
        if tank_storage[tank_num] > tank:
            overflows[tank_num] = tank_storage[tank_num] - tank
            tank_storage[tank_num] = tank
    fill_res = namedtuple("tank_overflows", ["tank_storage", "overflows"])
    return fill_res(tank_storage, overflows)


def rw_use(tank_storage, demand):
    use = demand
    tank_storage = tank_storage - demand
    if len(tank_storage[tank_storage < 0]) > 0:
        neg_vals = np.where(tank_storage < 0)
        for val in neg_vals:
            tank_storage[val] += demand[val]
            use[val] = tank_storage[val]
            tank_storage[val] = 0
    use_res = namedtuple("water_use", ["tank_storage", "rainW_use"])
    return use_res(tank_storage, use)


runtime = Timer()
runtime.start()
dt = 30
rain_dt = 60 * 10
beta = 5 / 4
manning = 0.012
sim_days = 1
sim_len = int(sim_days * 24 * 60 * 60 / dt)
t = np.linspace(0, sim_len, num=sim_len + 1)
t = t.astype(int)
hours = t * (dt / 60) / 60
days = hours / 24

tank_size = np.array([20, 20, 20])
tank_init_storage = np.array([20, 20, 20], dtype=np.longfloat)
tank_storage = tank_init_storage
roof = np.array([1000, 1000, 1000])
dwellers = np.array([150, 150, 150])

demand_dt = 3 * 60 * 60
demands_3h = np.array([5, 3, 20, 15, 12, 15, 18, 12])
demands_PD = 33
demands = np.array([])
for demand in demands_3h:
    demands = np.append(demands, np.ones(int(demand_dt / dt)) * (demand * (dt / demand_dt)))
demands = demands * demands_PD / 100
demand_volume = np.matmul(np.reshape(dwellers, (len(dwellers), 1)), np.reshape(demands, (1, len(demands)))) / 1000

tank_outlets = np.array([200, 200, 200])
tank_Ds = np.array([0.2, 0.2, 0.2])
tank_slopes = np.array([0.01, 0.01, 0.01])
tank_alphas = (0.501 / manning) * (tank_Ds ** (1 / 6)) * (tank_slopes ** 0.5)
c_tanks = tank_outlets / dt
outlet_max_A = 0.9 * (np.pi * ((tank_Ds / 2) ** 2))
outlet_max_Q = tank_alphas * (outlet_max_A ** beta)

pipes_L = np.array([500, 500, 500])
pipe_Ds = np.array([0.5, 0.5, 0.5])
pipe_slopes = np.array([0.01, 0.01, 0.01])
pipe_alphas = (0.501 / manning) * (pipe_Ds ** (1 / 6)) * (pipe_slopes ** 0.5)
c_pipes = pipes_L / dt

rain = np.zeros(int(sim_days * 24 * 3600 / rain_dt))
rainfile = '1hour-2.csv'
rain_input = np.genfromtxt(rainfile, delimiter=',')
rain[:len(rain_input)] = rain_input

'''
rain_10min = np.linspace(0, 0.3, 4)
rain_10min = np.append(rain_10min, np.flip(rain_10min))
rain_size = rain_dt / dt
rain = np.array([])
for rain_I in rain_10min:
    rain = np.append(rain, np.ones(int(rain_size)) * rain_I)
rain = np.append(np.zeros(int((sim_len - len(rain)) / 6)), rain)
rain = np.append(rain, np.zeros(sim_len - len(rain)))
rain[300:300+max(np.shape(np.nonzero(rain)))]=rain[np.nonzero(rain)]*1.2
'''
# rain_volume = np.matmul(np.reshape(roof, (len(roof), 1)), np.reshape(rain, (1, len(rain)))) / 1000
overflows = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
rainW_use = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
tank_storage_all = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
rain_volume = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
outlet_A = np.zeros((len(tank_outlets), sim_len, 2), dtype=np.longfloat)
outlet_Q = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
pipe_A = np.zeros((len(pipes_L), sim_len, 2), dtype=np.longfloat)
pipe_Q = np.zeros((len(pipes_L), sim_len, 2), dtype=np.longfloat)

for i in range(sim_len):
    current_rain = rain[int(i // (rain_dt / dt))]
    if sum(tank_storage) == 0 and sum(rain[int(i // (rain_dt / dt)):-1]) == 0:
        break
    rain_volume[:, i] = current_rain * (dt/rain_dt) * roof / 1000
    if np.sum(rain_volume[:, i]) > 0:
        fill_result = tank_fill(tank_storage, rain_volume[:, i], tank_size)
        tank_storage = fill_result.tank_storage
        overflows[:, i] = fill_result.overflows
    use_result = rw_use(tank_storage, demand_volume[:, i % demand_volume.shape[1]])
    tank_storage = use_result.tank_storage
    rainW_use[:, i] = use_result.rainW_use
    tank_storage_all[:, i] = tank_storage
    outlet_A[:, i, 0] = ((overflows[:, i] / dt) / tank_alphas) ** (1 / beta)
    if i < 1 or (np.sum(pipe_A[:, i - 1, :]) + np.sum(outlet_A[:, i - 1])) < 1e-5:
        continue
    for j in range(len(tank_outlets)):
        constants = tank_alphas[j] * beta * (dt / tank_outlets[j])
        outlet_A[j, i, 1] = outlet_A[j, i - 1, 1] - constants * (((outlet_A[j, i, 0] + outlet_A[j, i - 1, 1]) / 2.0) \
                                                                 ** (beta - 1)) * (
                                    outlet_A[j, i - 1, 1] - outlet_A[j, i, 0])
    outlet_Q[:, i] = tank_alphas * (outlet_A[:, i, 1] ** beta)
    for j in range(len(pipes_L)):
        if j > 0:
            pipe_Q[j, i, 0] = pipe_Q[j - 1, i, 1] + outlet_Q[j, i]
        else:
            pipe_Q[j, i, 0] = outlet_Q[j, i]
        pipe_A[j, i, 0] = (pipe_Q[j, i, 0] / pipe_alphas[j]) ** (1 / beta)
        constants = pipe_alphas[j] * beta * (dt / pipes_L[j])
        pipe_A[j, i, 1] = pipe_A[j, i - 1, 1] - constants * (
                ((pipe_A[j, i, 0] + pipe_A[j, i - 1, 1]) / 2) ** (beta - 1)) * \
                          (pipe_A[j, i - 1, 1] - pipe_A[j, i, 0])
        pipe_Q[j, i, 1] = pipe_alphas[j] * (pipe_A[j, i, 1] ** beta)

T = i
mass_balance_err = 100 * (abs(integrate.simps(pipe_Q[2, :, 1] * dt, t[0:-1]) - np.sum(overflows))) / np.sum(overflows)
print(f"Mass Balance Error: {mass_balance_err:0.2f}%")

max_Q = np.argmax(pipe_Q[2, :, 1])
zero_Q = (np.asarray(np.nonzero(pipe_Q[2, max_Q:-1, 1] < 1e-5))[0])[0] + max_Q
'''
plt.plot(hours[0:zero_Q+1], pipe_Q[2, :zero_Q+1, 1], label="optimized outlet flow")
plt.ylabel('Q (' + r'$m^3$' + '/s)')
plt.xlabel('t (hours)')
plt.legend()
'''

last_overflow = np.max(np.nonzero(np.sum(overflows, axis=0)))
obj_Q = integrate.simps(pipe_Q[2, :zero_Q, 1] * dt, t[:zero_Q]) / (last_overflow * dt)
to_min = float(0)
for i in range(last_overflow):
    to_min += abs(pipe_Q[2, i, 1] - obj_Q)
runtime.stop()
print(np.sum(rainW_use))
print(np.sum(tank_storage))
print(np.max(pipe_Q[2, :, 1]))
#print('_')
