import numpy as np
import datetime as dt

t1 = [3, 1, 2]
v1 = [6, 7, 8]
print(v1[np.argmin(t1)])

start_time = dt.datetime.now()
list = np.random.rand(1500000)
#print(list)
list = sorted(list)
#print(list)
end_time = dt.datetime.now()
times = (end_time - start_time)
print(times.seconds + times.microseconds / np.power(10, 6))
