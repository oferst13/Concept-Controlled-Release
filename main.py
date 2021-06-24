import numpy as np

def tank_fill(tank_storage, rain, tank_size):
    overflows = np.zeros_like(tank_size)
    for tank_num, tank in enumerate(tank_size):
        tank_storage[tank_num] = tank_storage[tank_num] + rain[tank_num]
        if tank_storage[tank_num] > tank:
            overflows[tank_num] = tank_storage[tank_num] - tank
            tank_storage[tank_num] = tank
    return tank_storage, overflows


dt = 30
rain_dt = 600
beta = 5/4
manning = 0.012
sim_len = 600
t = np.linspace(0, sim_len, num=sim_len+1)
t = t.astype(int)

tank_size = np.array([2, 2, 2])
tank_storage = np.zeros_like(tank_size)
roof = np.array([1000, 1000, 1000])

tank_outlets = np.array([500, 500, 500])
tank_Ds = np.array([0.2, 0.2, 0.2])
tank_slopes = np.array([0.02, 0.02, 0.02])
tank_alphas = (0.501/manning) * (tank_Ds ** (1/6)) * (tank_slopes ** 0.5)
c_tanks = tank_outlets / dt

pipes = np.array([1000, 1000, 1000])
pipe_Ds = np.array([0.5, 0.5, 0.5])
pipe_slopes = np.array([0.02, 0.02, 0.02])
pipe_alphas = (0.501/manning) * (pipe_Ds ** (1/6)) * (pipe_slopes ** 0.5)
c_pipes = pipes / dt

rain_10min = np.linspace(0, 0.6, 4)
rain_10min = np.append(rain_10min, np.flip(rain_10min))
rain_size = rain_dt / dt
rain = np.array([])
for rain_I in rain_10min:
    rain = np.append(rain, np.ones(int(rain_size)) * rain_I)

tank_storage, overflows = tank_fill(tank_storage, rain, tank_size)
print(overflows)




