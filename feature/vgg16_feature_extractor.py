import numpy as np
import cv2
from keras.applications import VGG16
from keras.applications import imagenet_utils
from feature.feature_extractor import FeatureExtractor


class VGG16FeatureExtractor(FeatureExtractor):
    """
    VGG16特征提取器

    es 2018-11-06
    """

    def __init__(self):
        """
        初始化
        """
        FeatureExtractor.__init__(self)

        # 定义参数
        self.width = 224
        self.height = 224
        self.channels = 3
        # 输出特征维度
        self.final_pools = 7 * 7 * 512

        # 加载预先已训练的模型
        self.model = VGG16(weights="imagenet", include_top=False)
        print("[INFO] loaded model ok: VGG16 + imagenet")

    def preprocess(self, imgs):
        """
        图像预处理
        """
        imgs_p = []
        for img in imgs:
            if img.shape[0] != self.height or img.shape[1] != self.width:
                img = cv2.resize(img, (self.width, self.height))
            img = np.expand_dims(img, axis=0)
            img = imagenet_utils.preprocess_input(img)
            imgs_p.append(img)
        imgs_p = np.vstack(imgs_p)
        return imgs_p

    def extract_features(self, imgs, batch_size=128, normalizing=False):
        """
        批量图像特征提取
        """
        # 预处理
        imgs_p = self.preprocess(imgs)

        # 分批处理
        imgs_f = []
        index = 0
        imgs_size = len(imgs)
        while index < imgs_size:
            index_end = index + batch_size
            if index_end > imgs_size:
                index_end = imgs_size

            # 向前流转网络，得到特征
            imgs_f_ = self.model.predict(imgs_p[index: index_end], batch_size=batch_size)
            # 将每个特征平铺
            imgs_f_ = imgs_f_.reshape((imgs_f_.shape[0], self.final_pools))
            # 将特征转为单位向量
            if normalizing:
                imgs_f_ = imgs_f_ / np.linalg.norm(imgs_f_)
            imgs_f.extend(imgs_f_)
            print("[INFO] extract_features {} - {} ok".format(index, index_end))

            index += batch_size

        return imgs_f
