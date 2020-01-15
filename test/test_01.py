import numpy as np
import datetime as dt

tds = np.random.rand(10000, 512)
datas = np.random.rand(10, 512)

dt0 = dt.datetime.now()

for data in datas:
    ds = []
    for i, td in enumerate(tds):
        #dist = np.linalg.norm(np.subtract(data, td))

        d = np.subtract(data, td)
        ds.append(d)

    '''data_ = data * np.ones((10000, 512))
    ds = tds - data_'''

dt1 = dt.datetime.now()

print(dt1 - dt0)
