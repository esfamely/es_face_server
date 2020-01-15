# import the necessary packages
import numpy as np
from sklearn.preprocessing import LabelEncoder
import cv2
import os
from pathlib import Path


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_dir, verbose=-1, one_hot=False):
        # initialize the list of features and labels
        data = []
        labels = []

        # loop over the input images
        image_path = Path(image_dir)
        list_image = list(image_path.glob("**/*.jpg"))
        for image in list_image:
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image_path = str(image)
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            # check to see if our preprocessors are not None
            if self.preprocessors is not None:
                # loop over the preprocessors and apply each to
                # the image
                for p in self.preprocessors:
                    image = p.preprocess(image)

            # treat our processed image as a "feature vector"
            # by updating the data list followed by the labels
            data.append(image)
            labels.append(label)

            # show an update every `verbose` images
            i = len(labels) - 1
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(list_image)))

        # encode the labels as integers
        le = LabelEncoder()
        labels = le.fit_transform(labels)

        if one_hot:
            labels_oh = np.zeros([len(labels), np.max(labels) + 1], dtype=np.int32)
            i = 0
            for label in labels:
                labels_oh[i][label - 1] = 1
                i += 1
            labels = labels_oh

        # return a tuple of the data and labels
        return np.array(data), np.array(labels), le
