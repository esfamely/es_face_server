class FeatureExtractor:
    """
    特征提取器

    es 2018-11-06
    """

    def __init__(self):
        """
        初始化
        """

    def preprocess(self, imgs):
        """
        图像预处理
        """

    def extract_feature(self, img):
        """
        一个图像特征提取
        """
        imgs = [img]
        imgs_f = self.extract_features(imgs)
        img_f = imgs_f[0]
        return img_f

    def extract_features(self, imgs):
        """
        批量图像特征提取
        """
