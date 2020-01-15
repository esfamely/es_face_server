import cv2


class HistPreprocessor:
    """
    数据预处理-直方图均衡化
    """
    def __init__(self, gray=False):
        # to gray ?
        self.gray = gray

    def preprocess(self, image):
        if self.gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = self.equalizeHist(image)

        return image

    def equalizeHist(self, image):
        """
        直方图均衡化
        """
        if image.ndim < 3:
            # 灰度图直接调用均衡化API
            image = cv2.equalizeHist(image)
        else:
            # 将彩色三通道图像转换到YUV颜色空间
            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

            # 将Y通道图像直方图均衡化
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])

            # 将YUV图像转换回RGB图像
            cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR, image)

        return image
