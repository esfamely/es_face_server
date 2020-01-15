"""
计算机视觉工具
"""
import numpy as np
import cv2
from pathlib import Path


def distance_l2(img1, img2):
    """
    计算两图像的L2距离
    """
    return np.sqrt(np.sum(np.square(np.subtract(img1, img2))))


def distance_l2_relative(img1, img2):
    """
    计算两图像的L2相对距离
    """
    distance = np.mean(np.square(img1 - img2))
    return distance


def area_rect(rect):
    """
    计算矩形面积，rect为左上角顶点坐标 + 宽高
    """
    return rect[2] * rect[3]


def area_box(box):
    """
    计算矩形面积，box为左上角顶点坐标 + 右下角顶点坐标
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def max_rect(rects):
    """
    找到最大的矩形
    """
    list = [area_rect(rect) for rect in rects]
    max_index = np.argmax(list)
    return rects[max_index]


def max_box(boxes):
    """
    找到最大的矩形
    """
    list = [area_box(box) for box in boxes]
    max_index = np.argmax(list)
    return boxes[max_index]


def list_imgs_files(dir):
    imgs = []
    files = []
    lFile = list(Path(dir).glob("*.jpg"))
    for file in lFile:
        img = cv2.imread(str(file))
        imgs.append(img)
        files.append(file.name)
    return imgs, files


def add_border(img, border=15, color=255):
    """
    图像增加纯色边框
    """
    width, height = img.shape[1], img.shape[0]
    img_border = np.ones((height + border * 2, width + border * 2, 3), np.uint8) * color
    img_border[border: border + height, border: border + width, :] = img
    return img_border
