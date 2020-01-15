import numpy as np
import tensorflow as tf
from dataset.mtcnn.detect_face import create_mtcnn, detect_face
from main.setup import Setup


class Mtcnn:
    """
    mtcnn人脸检测器
    """

    def __init__(self):
        # 加载mtcnn检测网络
        sess = tf.Session()
        with sess.as_default():
            self.pnet, self.rnet, self.onet = create_mtcnn(sess, None)

    def detect_face(self, img):
        """
        人脸检测
        :return: 人脸区域坐标与五个特征点坐标
        """
        # 最小检测尺寸
        minsize = Setup.s3_minsize
        # 三个cnn的检测阀值
        threshold = Setup.s3_threshold
        # 尺度因子
        factor = Setup.s3_factor

        total_boxes, points = detect_face(img, minsize, self.pnet, self.rnet, self.onet,
                                          threshold, factor)
        # 转置使得五个特征点坐标维度与人脸区域坐标的一致
        points = np.asarray(points).transpose()

        return total_boxes, points
