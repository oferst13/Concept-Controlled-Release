import numpy as np


length = 2
begin_zeros = 3
dt = 600
dt_per_hr = int(3600/dt)
rain_hour = 33.7
to_write = np.zeros(((begin_zeros + length)*dt_per_hr + 1), dtype=float)
to_write[begin_zeros*dt_per_hr:(begin_zeros+length)*dt_per_hr]=rain_hour/dt_per_hr
np.savetxt('2hour-1.csv', to_write, fmt='%1.4f', delimiter=',')
