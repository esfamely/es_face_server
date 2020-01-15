import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GaussianMixture
from scipy import stats
import cv2


def show_mg():
    """
    演示多维高斯分布图
    """
    mean = [0, 0]
    covariance = [[3, 0.5],
                  [0.5, 1]]
    mg = stats.multivariate_normal(mean, covariance)

    img = np.zeros((500, 500)).astype(np.float64)

    for r in range(-200, 201):
        for c in range(-200, 201):
            p = mg.pdf([r / 50, c / 50])
            img[r + 250, c + 250] = p

    img_max = np.max(img)
    img = (img * 255 / img_max).astype(np.uint8)

    cv2.imshow("es", img)
    cv2.waitKey(0)


def ppd(mog, x):
    """
    计算Pr(h=k|x,θ)
    """
    mgs = []
    for j in range(len(mog.weights_)):
        mg = stats.multivariate_normal(mog.means_[j], mog.covariances_[j])
        mgs.append(mg)

    p_xh = []
    for j in range(len(mog.weights_)):
        p_x_h = mgs[j].pdf(x)
        p_h = mog.weights_[j]
        p_xh.append(p_x_h * p_h)
    p_x = np.sum(p_xh)

    p_h_x = []
    for j in range(len(mog.weights_)):
        p_h_x.append(p_xh[j] / p_x)

    return p_h_x


def test1():
    X, y_true = make_blobs(n_samples=50, centers=4, cluster_std=1.60, random_state=10)

    #plt.scatter(X[:, 0], X[:, 1], s=15, cmap='viridis')
    #plt.show()

    mog = GaussianMixture(n_components=4).fit(X)
    labels = mog.predict(X)
    probs = mog.predict_proba(X)
    #print(mog.weights_)
    #print(mog.means_)
    #print(mog.covariances_)

    for i in range(50):
        p_h_x = ppd(mog, X[i])
        print("{}, {}, {}, {}".format(labels[i], np.argmax(p_h_x), probs[i], p_h_x))

    X_add_mean = np.concatenate([X, mog.means_])
    labels_add_mean = np.concatenate([labels, np.ones(4) * 4])

    plt.scatter(X_add_mean[:, 0], X_add_mean[:, 1], c=labels_add_mean, s=25, cmap='viridis')
    plt.show()


def test2():
    img = cv2.imread("D:/s5/lena/n01.jpg")

    X = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            X.append(np.concatenate(([r, c], img[r, c])))
            #print(X[-1])

    n_components = 3
    mog = GaussianMixture(n_components=n_components).fit(X)
    labels = mog.predict(X)
    #print(labels)

    index = 0
    img_seg = np.zeros_like(img)
    for r in range(img_seg.shape[0]):
        for c in range(img_seg.shape[1]):
            pv = int(255 * labels[index] / (n_components - 1))
            img_seg[r][c] = [pv, pv, pv]
            index += 1

    cv2.imshow("es", cv2.resize(img_seg, (img_seg.shape[1] * 3, img_seg.shape[0] * 3)))
    cv2.waitKey(0)


#show_mg()
#test1()
test2()
