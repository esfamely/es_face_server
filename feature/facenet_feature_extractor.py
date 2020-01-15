import cv2
import tensorflow as tf
import feature.facenet.facenet as facenet
from feature.feature_extractor import FeatureExtractor
from utils.utils_sys import info


class FacenetFeatureExtractor(FeatureExtractor):
    """
    Facenet特征提取器

    es 2018-11-06
    """

    def __init__(self, model_path):
        """
        初始化
        """
        FeatureExtractor.__init__(self)

        # 定义参数
        self.width = 160
        self.height = 160
        self.channels = 3

        # 加载预先已训练的模型
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        加载预先已训练的模型
        """
        sess = tf.Session()

        facenet.load_model(model_path)
        info("loaded model ok: %s" % model_path)

        # 获取输入与输出tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # 记录已加载模型的会话
        self.sess = sess

        # 记录输入与输出tensors
        self.images_placeholder = images_placeholder
        self.embeddings = embeddings
        self.phase_train_placeholder = phase_train_placeholder

    def preprocess(self, imgs):
        """
        图像预处理
        """
        imgs_p = []
        for img in imgs:
            if img.shape[0] != self.height or img.shape[1] != self.width:
                img = cv2.resize(img, (self.width, self.height))
            img_p = facenet.prewhiten(img)
            imgs_p.append(img_p)
        return imgs_p

    def extract_features(self, imgs, batch_size=256):
        """
        批量图像特征提取
        """
        # 预处理
        imgs_p = self.preprocess(imgs)

        # 获取已加载模型的会话
        sess = self.sess

        # 分批处理
        imgs_f = []
        index = 0
        imgs_size = len(imgs)
        while index < imgs_size:
            index_end = index + batch_size
            if index_end > imgs_size:
                index_end = imgs_size

            # 向前流转网络，得到特征
            imgs_f_ = sess.run(self.embeddings,
                               feed_dict={self.images_placeholder: imgs_p[index: index_end],
                                          self.phase_train_placeholder: False})
            imgs_f.extend(imgs_f_)
            info("extract_features {} - {} ok".format(index, index_end))
            index += batch_size

        return imgs_f
