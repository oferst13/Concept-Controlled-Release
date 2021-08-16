import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from benchmark import obj_Q, zero_Q, outlet_max_Q, tank_fill, rw_use, pipe_Q as benchmark_Q
import math
from linetimer import CodeTimer

with CodeTimer():
    X = [5.,  4.,  2.,  9.,  6.,  7.,  0., 10.,  2., 10.,  1.,  3.,  9.,  5.,  7.,  9.,  4.,  3.,
        4., 10.,  5.,  3.,  5.,  6.,  4.,  6.,  8.,  1.,  8.,  8.,  8.,  0.,  7.,  1.,  1.,  9.]
    release = np.array(X)
    dt = 30
    rain_dt = 600
    beta = 5 / 4
    manning = 0.012
    # sim_len = (60 / dt) * 24
    sim_days = 3
    sim_len = int(sim_days * 24 * 60 * 60 / dt)
    t = np.linspace(0, sim_len, num=sim_len + 1)
    t = t.astype(int)
    hours = t * (dt / 60) / 60
    days = hours / 24

    tank_size = np.array([20, 20, 20])
    tank_storage = np.array([20, 20, 20], dtype=np.longfloat)
    roof = np.array([1000, 1000, 1000])
    dwellers = np.array([150, 150, 150])

    release_hrs = math.ceil(zero_Q * (dt / 60) / 60)
    release = release.reshape((len(tank_storage), release_hrs), order='F')
    tank_orifice = np.array([0.05, 0.05, 0.05])
    tank_orifice_A = ((tank_orifice / 2) ** 2) * np.pi
    Cd = 0.6
    tank_D = np.array([2.8, 2.8, 2.8])
    tank_A = ((tank_D / 2) ** 2) * np.pi
    releases_volume = np.zeros((len(tank_storage), sim_len), dtype=np.longfloat)

    demand_dt = 3 * 60 * 60
    demands_3h = np.array([5, 3, 20, 15, 12, 15, 18, 12])
    demands_PD = 33
    demands = np.array([])
    for demand in demands_3h:
        demands = np.append(demands, np.ones(int(demand_dt / dt)) * (demand * (dt / demand_dt)))
    demands = demands * demands_PD / 100
    demand_volume = np.matmul(np.reshape(dwellers, (len(dwellers), 1)), np.reshape(demands, (1, len(demands)))) / 1000

    tank_outlets = np.array([500, 500, 500])
    tank_Ds = np.array([0.2, 0.2, 0.2])
    tank_slopes = np.array([0.02, 0.02, 0.02])
    tank_alphas = (0.501 / manning) * (tank_Ds ** (1 / 6)) * (tank_slopes ** 0.5)
    c_tanks = tank_outlets / dt
    outlet_max_A = 0.9 * (np.pi * ((tank_Ds / 2) ** 2))
    outlet_max_Q = tank_alphas * (outlet_max_A ** beta)

    pipes_L = np.array([1000, 1000, 1000])
    pipe_Ds = np.array([0.5, 0.5, 0.5])
    pipe_slopes = np.array([0.02, 0.02, 0.02])
    pipe_alphas = (0.501 / manning) * (pipe_Ds ** (1 / 6)) * (pipe_slopes ** 0.5)
    c_pipes = pipes_L / dt

    rain_10min = np.linspace(0, 0.3, 4)
    rain_10min = np.append(rain_10min, np.flip(rain_10min))
    rain_size = rain_dt / dt
    rain = np.array([])
    for rain_I in rain_10min:
        rain = np.append(rain, np.ones(int(rain_size)) * rain_I)
    rain = np.append(np.zeros(int((sim_len - len(rain)) / 5)), rain)
    rain = np.append(rain, np.zeros(sim_len - len(rain)))
    rain_volume = np.matmul(np.reshape(roof, (len(roof), 1)), np.reshape(rain, (1, len(rain)))) / 1000
    overflows = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)

    rainW_use = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
    tank_storage_all = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)

    outlet_A = np.zeros((len(tank_outlets), sim_len, 2), dtype=np.longfloat)
    outlet_Q = np.zeros((len(tank_outlets), sim_len), dtype=np.longfloat)
    pipe_A = np.zeros((len(pipes_L), sim_len, 2), dtype=np.longfloat)
    pipe_Q = np.zeros((len(pipes_L), sim_len, 2), dtype=np.longfloat)

    to_min = 0
    penalty = 0
    runs = 0

    for i in range(zero_Q):
        if sum(tank_storage) == 0 and sum(rain[i:-1]) == 0:
            break
        runs += 1
        if np.sum(rain_volume[:, i]) > 0:
            fill_result = tank_fill(tank_storage, rain_volume[:, i], tank_size)
            tank_storage = fill_result.tank_storage
            overflows[:, i] = fill_result.overflows
        if i < zero_Q:
            release_Q = tank_orifice_A * Cd * np.sqrt(2 * 9.81 * (tank_storage / tank_A)) * 0.1 * release[:, i // int(
                60 * 60 / dt)]
        else:
            release_Q = 0
        releases_volume[:, i] = release_Q * dt
        tank_storage -= release_Q * dt
        if len(tank_storage[tank_storage < 0]) > 0:
            neg_vals = np.where(tank_storage < 0)
            for val in neg_vals:
                tank_storage[val] += releases_volume[val, i]
                releases_volume[val, i] = tank_storage[val]
                tank_storage[val] = 0.00
        use_result = rw_use(tank_storage, demand_volume[:, i % demand_volume.shape[1]])
        tank_storage = use_result.tank_storage
        rainW_use[:, i] = use_result.rainW_use
        tank_storage_all[:, i] = tank_storage
        outlet_A[:, i, 0] = np.power((((overflows[:, i] + release_Q * dt) / dt) / tank_alphas), (1 / beta))
        if i < 1 or (np.sum(pipe_A[:, i - 1, :]) + np.sum(outlet_A[:, i - 1])) < 1e-6:
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

        if i < zero_Q:
            to_min += abs(pipe_Q[2, i, 1] - obj_Q)

    # line_objects = plt.plot(hours[0:zero_Q+1], np.transpose(pipe_Q[2, 0:zero_Q+1, 1], np.transpose(benchmark_Q[2, 0:zero_Q+1, 1])))
    # plt.gca().set_prop_cycle(None)
    # line_objects.extend(plt.plot(hours[0:-1], np.transpose(pipe_Q[:, :, 0]), '-.', linewidth=1))

    plt.plot(hours[0:zero_Q + 100], pipe_Q[2, :zero_Q + 100, 1], label="optimized outlet flow")
    plt.plot(hours[0:zero_Q + 100], benchmark_Q[2, :zero_Q + 100, 1], label="benchmark outlet flow")
    plt.ylabel('Q (' + r'$m^3$' + '/s)')
    plt.xlabel('t (hours)')
    plt.legend()
    # plt.legend(line_objects, ('Pipe 1 - outflow', 'Pipe 2 - outflow', 'Pipe 3 - outflow', 'Pipe 1 - inflow', \
    # 'Pipe 2 - inflow', 'Pipe 3 - inflow'))

    mass_balance_err = 100 * (
                abs(integrate.simps(pipe_Q[2, :, 1] * dt, t[0:-1]) - (np.sum(overflows) + np.sum(releases_volume))) /
                (np.sum(overflows) + np.sum(releases_volume)))
    print(f"Mass Balance Error: {mass_balance_err:0.2f}%")
    to_min += penalty
    print('_')

'''
varbound = np.array([[0, obj_Q*2]] * 6)
algorithm_param = {'max_num_iteration': 100,\
                   'population_size':50,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':100}
model = ga(function=f, dimension=6, variable_type='real', variable_boundaries=varbound,algorithm_parameters=algorithm_param, function_timeout=15)
model.run()
'''
