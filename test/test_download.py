import os
import time
import socket
from utils.utils_web import download_file

socket.setdefaulttimeout(60)

for i in range(1, 181):
    img_path = "http://p.9090rt.info/uploadfile/2019/0814/05/{:02d}.jpg".format(i)
    path = "D:/pyimg/123/{:03d}.jpg".format(i)
    if os.path.exists(path) is False:
        rp = download_file(img_path, path)
        if rp is None:
            print(img_path)
        else:
            print(rp)
        time.sleep(1.5)
