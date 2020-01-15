import numpy as np
import cv2

img = cv2.imread("D:/s5/lena/AA5125.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("D:/s5/lena/AA5125_.jpg", img_gray)

kernel = np.asarray([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]])
img_edge = cv2.filter2D(img_gray, -1, kernel)

img_edge = cv2.dilate(img_edge, np.ones((3, 3)))
img_edge = cv2.erode(img_edge, np.ones((3, 3)))

_, contours, hierarchy = cv2.findContours(img_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(contours):
    print(i)
    cv2.drawContours(img, contours, i, (0, 0, 255), thickness=5)

# 直方个数
hist_size = 256
# 统计值范围
hist_range = [0, 255]
# 直方图
hist = cv2.calcHist([img_edge], [0], np.asarray([]), [hist_size], hist_range)
#for i, h in enumerate(hist):
#    print("{}: {}".format(i, h))

img_show = img
cv2.imwrite("D:/s5/lena/AA5125_0.jpg", img_show)
