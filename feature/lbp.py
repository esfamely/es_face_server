"""
LBP特征提取
"""

import numpy as np
import cv2

PI = 3.14159

# 邻域圈半径
LBP_RADIUS = 1
# 邻域个数
LBP_NEIGHBORS = 8
# 分块数（横向）
LBP_GRID_X = 8
# 分块数（纵向）
LBP_GRID_Y = 8

# 从这个角度对应的邻域作为二进制首位
LBP_START_ANGLE = (3.0 / 4) * PI
# 统一化模式表最大二进制位数
LBP_UP_SIZE = 16


class Lbp:
    def __init__(self, radius=LBP_RADIUS, neighbors=LBP_NEIGHBORS,
                 grid_x=LBP_GRID_X, grid_y=LBP_GRID_Y):
        """
        :param radius: 邻域圈半径
        :param neighbors: 邻域个数
        :param grid_x: 分块数（横向）
        :param grid_y: 分块数（纵向）
        """
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y

    def lbp_img(self, img):
        """
        生成图像的LBP图像
        """
        img = np.asarray(img)

        # 不少于邻域圈半径的最小整数
        _radius = int(np.ceil(self.radius))
        _radiusX2 = _radius * 2

        # LBP图像不包括邻域圈半径的边界，并且插值需要右下方向像素，所以边界需再减一
        lbpImg = np.zeros((img.shape[0] - _radiusX2 - 1, img.shape[1] - _radiusX2 - 1),
                          dtype=np.uint8)

        # 每个邻域间隔角度
        angle = 2 * PI / self.neighbors

        # 扫描每个邻域
        for m in range(self.neighbors):
            # 当前邻域对应中心点的角度
            _angle = LBP_START_ANGLE - m * angle

            # 计算当前邻域位置
            nx = _radius + self.radius * np.cos(_angle)
            ny = _radius - self.radius * np.sin(_angle)

            # 逐行扫描
            for r in range(_radius, img.shape[0] - _radius - 1):
                # 逐列扫描
                for c in range(_radius, img.shape[1] - _radius - 1):
                    # 当前邻域跟随着移动
                    _nx = nx + c - _radius
                    _ny = ny + r - _radius

                    # 当前邻域点周边（左上，左下，右上，右下）四个点位置
                    x1, y1 = int(_nx), int(_ny)
                    x2, y2 = x1, y1 + 1
                    x3, y3 = x1 + 1, y1
                    x4, y4 = x1 + 1, y1 + 1

                    # 四个点像素
                    p1 = img[y1][x1]
                    p2 = img[y2][x2]
                    p3 = img[y3][x3]
                    p4 = img[y4][x4]

                    # 双线性插值，计算当前邻域点像素
                    dx = _nx - np.floor(_nx)
                    dy = _ny - np.floor(_ny)
                    v1 = (1 - dx) * p1 + dx * p3
                    v2 = (1 - dx) * p2 + dx * p4
                    v = (1 - dy) * v1 + dy * v2

                    # 当前中心点像素
                    cp = img[r][c]

                    # 计算当前位（邻域）LBP值
                    b = 1 if (v - cp) >= 0 else 0
                    if b != 0:
                        lbp10 = 1 << (self.neighbors - m - 1)

                        # 累计LBP值
                        lbpImg[r - _radius, c - _radius] += lbp10

        return lbpImg

    def lbp_hist(self, lbpImg, minVal, maxVal):
        """
        生成LBP图像的统计直方图
        """
        # 直方个数
        histSize = maxVal - minVal + 1
        # 统计值范围
        range = [minVal, maxVal + 1]

        # 统计
        data = cv2.calcHist([lbpImg], [0], np.asarray([]), [histSize], range)

        # 转为一行数据
        #data = data.reshape(1, -1)

        return data

    def extract_feature(self, img):
        """
        LBP特征提取
        """
        # 生成图像的LBP图像
        lbpImg = self.lbp_img(img)
        lbpImg = np.asarray(lbpImg)

        # LBP二进制位数
        cc = int(np.power(2, LBP_NEIGHBORS))

        # 初始化特征向量，向量维度 = LBP模式数 * 总分块数
        dim = cc * self.grid_x * self.grid_y
        lbpVector = np.zeros(dim, dtype=np.float32)

        # 分块提取特征
        xcc = int(lbpImg.shape[1] / self.grid_x)
        ycc = int(lbpImg.shape[0] / self.grid_y)
        index = 0
        for m in range(self.grid_y):
            for n in range(self.grid_x):
                x, y, w, h = n*xcc, m*ycc, xcc, ycc
                if n == (self.grid_x - 1):
                    w = lbpImg.shape[1] - n * xcc
                if m == (self.grid_y - 1):
                    h = lbpImg.shape[0] - m * ycc
                subImg = lbpImg[y: y + h, x: x + w]

                # 直方图
                data = self.lbp_hist(subImg, 0, cc - 1)
                for i in range(len(data)):
                    lbpVector[index] = data[i] / (w * h)
                    index += 1

        return lbpVector

    def to_up(self, feature, ignore_noise=True):
        """
        转换为统一化模式
        """
        feature_up = []
        ups = []
        up_limit = 2

        # LBP二进制位数
        cc = int(np.power(2, LBP_NEIGHBORS))

        # 属于统一化模式的十进制数
        for m in range(cc):
            # 转成二进制数组
            bts = [b for b in "{:08b}".format(m)]
            # 相邻两位跳变计数
            tc = 0
            for i, b in enumerate(bts[0:-1]):
                if bts[i] != bts[i + 1]:
                    tc += 1
                    if tc > up_limit:
                        break
            is_up = tc <= up_limit
            if is_up:
                ups.append(m)

        num = int(len(feature) / cc)
        for m in range(num):
            feature_t = feature[m: m + cc]
            nupc = 0
            for n in range(len(feature_t)):
                if ups.count(n) > 0:
                    feature_up.append(feature_t[n])
                else:
                    nupc += feature_t[n]
            if not ignore_noise:
                feature_up.append(nupc)

        return feature_up


'''lbp = Lbp()
img_in = cv2.imread("D:/s5/lena/lbp/tt1.jpg")
img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
feature = lbp.extract_feature(img_in)
print(len(feature))

feature = feature
feature_up = lbp.to_up(feature, ignore_noise=False)
print(len(feature_up))
for i in range(len(feature_up)):
    print(str(i) + ": " + str(feature_up[i]))'''
