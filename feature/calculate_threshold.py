import numpy as np


def calculate_accuracy(ds, ls, threshold):
    """
    通过阀值分类并计算准确率
    """
    ds_t = np.greater_equal(ds, threshold).astype(np.int32)
    es = np.equal(ds_t, ls).astype(np.int32)
    acc = np.mean(es)
    return acc


def statistics_from_train(ds, ls):
    """
    从训练数据集计算两类样本的距离均值
    """
    ds1, ds2 = [], []
    for i, d in enumerate(ds):
        if ls[i] == 0:
            ds1.append(d)
        else:
            ds2.append(d)
    mean1 = np.mean(ds1)
    mean2 = np.mean(ds2)
    return mean1, mean2


def calculate_threshold_from_vat(ds, ls, mean1, mean2, num=100):
    """
    从验证数据集计算最佳分类阀值
    """
    center = (mean1 + mean2) / 2
    width = np.abs(mean1 - mean2)
    range = width / 4
    thresholds = np.linspace(center - range, center + range, num)
    acc_max = 0
    threshold_best = 0
    for threshold in thresholds:
        acc = calculate_accuracy(ds, ls, threshold)
        #print("{} - {}".format(threshold, acc))
        if acc > acc_max:
            acc_max = acc
            threshold_best = threshold
    return threshold_best, acc_max, range


def show_from_test(ds, ls, threshold, range, num=100):
    """
    显示在测试数据集上的准确率
    """
    thresholds = np.linspace(threshold - range, threshold + range, num)
    acc_max = 0
    threshold_best = 0
    for threshold in thresholds:
        acc = calculate_accuracy(ds, ls, threshold)
        print("{} - {}".format(threshold, acc))
        if acc > acc_max:
            acc_max = acc
            threshold_best = threshold
    return threshold_best, acc_max


'''ds = [1, 2, 3, 4.5, 5.5, 4, 5, 6, 7, 8, 9, 10]
ls = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

mean1, mean2 = statistics_from_train(ds, ls)
print("{} - {}".format(mean1, mean2))
threshold_best_vat, acc_max_vat, range = calculate_threshold_from_vat(ds, ls, mean1, mean2, num=10)
print("best_vat: {} - {}".format(threshold_best_vat, acc_max_vat))
threshold_best, acc_max = show_from_test(ds, ls, threshold_best_vat, range, num=10)
print("best: {} - {}".format(threshold_best, acc_max))'''
