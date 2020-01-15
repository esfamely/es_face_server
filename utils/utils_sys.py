"""
系统常用工具
"""
import numpy as np
import datetime as dt
import os
from pathlib import Path


def id_generator():
    dt_str = dt.datetime.now().strftime("%Y%m%d%H%M%S%f")
    id = dt_str
    id += str(np.random.rand(1)[0])[3:15]
    return id


def now_dt_str(format="%Y-%m-%d %H:%M:%S"):
    dt_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return dt_str


def info(txt):
    dt_str = now_dt_str()
    print("[INFO][" + dt_str + "]" + txt)


def del_files(dir, file_type="*.*"):
    lFile = list(Path(dir).glob(file_type))
    for file in lFile:
        os.remove(file.absolute())
