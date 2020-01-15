"""
web常用工具
"""
from flask import request
import urllib.request
import os


def getParameter(key, default=None):
    value = default
    try:
        value = request.form[key]
        return value
    except:
        value = default
    try:
        value = request.args[key]
        return value
    except:
        value = default
    try:
        value = request.get_json()[key]
        return value
    except:
        value = default
    return value


def getFile(key):
    try:
        file = request.files[key]
        return file
    except:
        return None


def download_file(url, path):
    user_agent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko)"
    user_agent += " Chrome/57.0.2987.98 Safari/537.36 LBBROWSER"
    header = {"User-Agent": user_agent}

    f = None
    try:
        f = open(path, 'wb')
        request = urllib.request.Request(url, headers=header)
        response = urllib.request.urlopen(request, timeout=100)
        f.write(response.read())
        response.close()
        f.close()
        return path
    except:
        f.close()
        if os.path.exists(path) == True:
            os.remove(path)
        return None
