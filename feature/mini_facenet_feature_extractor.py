import numpy as np
import cv2
import tensorflow as tf
from feature.model_mini_facenet import MiniFacenet
from feature.feature_extractor import FeatureExtractor


class MiniFacenetFeatureExtractor(FeatureExtractor):
    """
    MiniFacenet特征提取器

    es 2018-11-06
    """

    def __init__(self, model_path):
        """
        初始化
        """
        FeatureExtractor.__init__(self)

        # 定义参数
        self.width = 40
        self.height = 40
        self.channels = 3

        # 加载预先已训练的模型
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        加载预先已训练的模型
        """
        with tf.variable_scope('placeholder'):
            Img = tf.placeholder("float", [None, self.width, self.height, self.channels])
            Dropout = tf.placeholder(tf.float32)
            Is_Training = tf.placeholder("bool")

        with tf.variable_scope('loss'):
            mini_facenet = MiniFacenet(self.channels)
            img_f = mini_facenet.conv_net(Img, self.width, self.height, self.channels,
                                          dropout=Dropout, is_training=Is_Training)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore model weights from previously saved model
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print("[INFO] loaded model ok: %s" % model_path)

        # 记录已加载模型的会话
        self.sess = sess
        self.graphs = {
            "Img": Img,
            "Dropout": Dropout,
            "Is_Training": Is_Training,
            "img_f": img_f
        }

    def preprocess(self, imgs):
        """
        图像预处理
        """
        imgs_p = []
        for img in imgs:
            if img.shape[0] != self.height or img.shape[1] != self.width:
                #print("resize_mini")
                img = cv2.resize(img, (self.width, self.height))
            imgs_p.append(img)
        imgs_p = np.stack(imgs_p)
        imgs_p = imgs_p / 255
        return imgs_p

    def extract_features(self, imgs, batch_size=256):
        """
        批量图像特征提取
        """
        # 预处理
        imgs_p = self.preprocess(imgs)

        # 获取已加载模型的会话
        sess = self.sess
        graphs = self.graphs

        # 分批处理
        imgs_f = []
        index = 0
        imgs_size = len(imgs)
        while index < imgs_size:
            index_end = index + batch_size
            if index_end > imgs_size:
                index_end = imgs_size

            # 向前流转网络，得到特征
            imgs_f_ = sess.run(graphs["img_f"], feed_dict={graphs["Img"]: imgs_p[index : index_end],
                                                           graphs["Dropout"]: 1.0,
                                                           graphs["Is_Training"]: False})
            imgs_f.extend(imgs_f_)
            print("[INFO] extract_features {} - {} ok".format(index, index_end))

            index += batch_size

        return imgs_f
