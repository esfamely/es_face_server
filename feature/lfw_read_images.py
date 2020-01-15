import numpy as np
import cv2
import os


class LfwReadImages:
    @staticmethod
    def read_images(dataset_path, pairs_path):
        img_s1, img_s2, labels = [], [], []
        file_format = "{}/{}_{:04d}.png"

        pairs = open(pairs_path).read().strip().split("\n")
        pairs = [r.split("\t") for r in pairs]
        for i, pair in enumerate(pairs):
            # 同一个人的脸
            if len(pair) == 3:
                img_path1 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[1])))
                img_path2 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[2])))
                if os.path.exists(img_path1) and os.path.exists(img_path2):
                    img1 = cv2.imread(img_path1)
                    img2 = cv2.imread(img_path2)
                    img_s1.append(img1)
                    img_s2.append(img2)
                    labels.append(0)
            # 不同人的脸
            if len(pair) == 4:
                img_path1 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[1])))
                img_path2 = os.path.join(dataset_path,
                                         file_format.format(pair[2], pair[2], int(pair[3])))
                if os.path.exists(img_path1) and os.path.exists(img_path2):
                    img1 = cv2.imread(img_path1)
                    img2 = cv2.imread(img_path2)
                    img_s1.append(img1)
                    img_s2.append(img2)
                    labels.append(1)

        return img_s1, img_s2, labels

    @staticmethod
    def make_triplet(dataset_path, pairs_path):
        """
        读取图像三元组，维持原有的组合
        """

        # 遍历一次，记录同人与不同人信息
        list1, list2 = [], []
        file_format = "{}/{}_{:04d}.png"
        pairs = open(pairs_path).read().strip().split("\n")
        pairs = [r.split("\t") for r in pairs]
        for i, pair in enumerate(pairs):
            # 同一个人的脸
            if len(pair) == 3:
                img_path1 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[1])))
                img_path2 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[2])))
                if os.path.exists(img_path1) and os.path.exists(img_path2):
                    map1 = {"name": pair[0], "img1": img_path1, "img2": img_path2}
                    list1.append(map1)
            # 不同人的脸
            if len(pair) == 4:
                img_path1 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[1])))
                img_path2 = os.path.join(dataset_path,
                                         file_format.format(pair[2], pair[2], int(pair[3])))
                if os.path.exists(img_path1) and os.path.exists(img_path2):
                    map2 = {"name1": pair[0], "name2": pair[2],
                            "img1": img_path1, "img2": img_path2,
                            "used": 0}
                    list2.append(map2)

        # 记录图像三元组
        imgs_anchor, imgs_positive, imgs_negative = [], [], []
        # 以同人信息，翻查不同人信息，看是否可以组队
        for map1 in list1:
            for map2 in list2:
                if map2["used"] == 0 and map1["name"] == map2["name1"]:
                    imgs_anchor.append(map1["img1"])
                    imgs_positive.append(map1["img2"])
                    imgs_negative.append(map2["img2"])
                    map2["used"] = 1
                    break
                if map2["used"] == 0 and map1["name"] == map2["name2"]:
                    imgs_anchor.append(map1["img1"])
                    imgs_positive.append(map1["img2"])
                    imgs_negative.append(map2["img1"])
                    map2["used"] = 1
                    break

        '''for img1, img2, img3 in zip(imgs_anchor, imgs_positive, imgs_negative):
            print(img1.split("/")[-1]
                  + ", " + img2.split("/")[-1]
                  + ", " + img3.split("/")[-1])
        print(len(imgs_anchor))'''

        return imgs_anchor, imgs_positive, imgs_negative

    @staticmethod
    def make_triplet2(dataset_path, pairs_path, cell_size=10):
        """
        读取图像三元组，利用所有可能产生的组合
        """

        # 遍历一次，记录以下信息：
        # {'人名1': ['图片1', '图片2', ...], '人名2': ['图片1', '图片2', ...]}
        dict = {}
        list_pair = []
        file_format = "{}/{}_{:04d}.png"
        pairs = open(pairs_path).read().strip().split("\n")
        pairs = [r.split("\t") for r in pairs]
        for i, pair in enumerate(pairs):
            # 同一个人的脸
            if len(pair) == 3:
                img_path1 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[1])))
                img_path2 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[2])))
                if os.path.exists(img_path1) and os.path.exists(img_path2):
                    list = dict.get(pair[0])
                    if list is None:
                        dict[pair[0]] = [img_path1, img_path2]
                    else:
                        if list.count(img_path1) == 0:
                            list.append(img_path1)
                        if list.count(img_path2) == 0:
                            list.append(img_path2)

                    list_pair.append(img_path1)
                    list_pair.append(img_path2)
            # 不同人的脸
            if len(pair) == 4:
                img_path1 = os.path.join(dataset_path,
                                         file_format.format(pair[0], pair[0], int(pair[1])))
                img_path2 = os.path.join(dataset_path,
                                         file_format.format(pair[2], pair[2], int(pair[3])))
                if os.path.exists(img_path1) and os.path.exists(img_path2):
                    list1 = dict.get(pair[0])
                    if list1 is None:
                        dict[pair[0]] = [img_path1]
                    else:
                        if list1.count(img_path1) == 0:
                            list1.append(img_path1)

                    list2 = dict.get(pair[2])
                    if list2 is None:
                        dict[pair[2]] = [img_path2]
                    else:
                        if list2.count(img_path2) == 0:
                            list2.append(img_path2)

                    list_pair.append(img_path1)
                    list_pair.append(img_path2)

        # 排好顺序，便于观察
        for key in dict:
            list = dict[key]
            dict[key] = sorted(list)
        '''for key in dict:
            list = dict[key]
            list = [path.split("/")[-1] for path in list]
            print(list)'''

        # 记录图像三元组
        imgs_anchor, imgs_positive, imgs_negative = [], [], []
        for key in dict:
            list = dict[key]

            len_list = len(list)
            # 同一个人有多张图片的，两两组合，再与其他人图片组合
            if len_list > 1:
                for i in range(len_list):
                    for j in range(i + 1, len_list):
                        if cell_size <= 0:
                            # 与其他人所有图片组合
                            for key2 in dict:
                                if key2 != key:
                                    list2 = dict[key2]
                                    for path in list2:
                                        imgs_anchor.append(list[i])
                                        imgs_positive.append(list[j])
                                        imgs_negative.append(path)
                        else:
                            # 因为与其他人所有图片组合会产生超大量的样本，所以这里采用随机抽取几份
                            index = 0
                            while True:
                                rd = np.random.rand(1)[0]

                                path = list_pair[int(rd * len(list_pair))]
                                if path.split("/")[-2] == list[i].split("/")[-2]:
                                    # 抽到的是同一个人，跳过再重新抽
                                    #print("ko: {}".format(path.split("/")[-2]))
                                    continue

                                imgs_anchor.append(list[i])
                                imgs_positive.append(list[j])
                                imgs_negative.append(path)

                                index += 1
                                if index >= cell_size:
                                    break

        '''for img1, img2, img3 in zip(imgs_anchor, imgs_positive, imgs_negative):
            print(img1.split("/")[-1] + ", " + img2.split("/")[-1] + ", " + img3.split("/")[-1])
        print(len(imgs_anchor))'''

        return imgs_anchor, imgs_positive, imgs_negative


'''root_dir = "D:/s5/dataset/"
lfw_path = root_dir + "lfw_160/"
pairs_train_path = root_dir + "lfw_txt/pairsDevTrain.txt"
LfwReadImages.make_triplet2(lfw_path, pairs_train_path, cell_size=10)'''
