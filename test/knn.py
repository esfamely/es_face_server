import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from preprocessing.simple_preprocessor import SimplePreprocessor
from preprocessing.hist_preprocessor import HistPreprocessor
from dataset.simple_dataset_loader import SimpleDatasetLoader
from feature.lbp import Lbp
from feature.mini_facenet_feature_extractor import MiniFacenetFeatureExtractor
from feature.facenet_feature_extractor import FacenetFeatureExtractor
from feature.vgg16_feature_extractor import VGG16FeatureExtractor
from feature.resnet50_feature_extractor import ResNet50FeatureExtractor

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
image_dir = "D:/s5/cv_python/deep_learning/dataset/easyface"

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
n = 160
c = 1
sp = SimplePreprocessor(n, n)
hist = HistPreprocessor(gray=True)
sdl = SimpleDatasetLoader(preprocessors=[sp, hist])
(data, labels, le) = sdl.load(image_dir, verbose=500)
#data = data.reshape((data.shape[0], n*n*c))

# lbp extract feature
'''lbp = Lbp()
data_t = []
for i, d in enumerate(data):
    data_t.append(lbp.extract_feature(data[i]))
    if i % 10 == 0:
        print("[INFO] lbp extract feature: {}".format(i))
data = data_t'''

# 1 channel to 3 channels
data_t = []
for i, d in enumerate(data):
    data_t.append(np.dstack([d, d, d]))
data = data_t

# facenet extract feature
#mini_model_path = "D:/s5/dataset/lfw_txt/facenet/160_40_1_20_0.00.ckpt"
#feature_extractor = MiniFacenetFeatureExtractor(mini_model_path)
#model_path = "D:/s5/cv_python/facenet/models/20180402-114759/20180402-114759.pb"
#feature_extractor = FacenetFeatureExtractor(model_path)
#feature_extractor = VGG16FeatureExtractor()
feature_extractor = ResNet50FeatureExtractor()
data = feature_extractor.extract_features(data, batch_size=16, normalizing=True)

# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.5, random_state=10)

# train and evaluate a k-NN classifier on the raw pixel intensities
for i in range(1):
    print("[INFO] evaluating k-NN classifier (k = {}) ...".format(i + 1))
    model = KNeighborsClassifier(i + 1, n_jobs=-1)
    #model = LogisticRegression()
    model.fit(trainX, trainY)
    preds = model.predict(testX)
    print(classification_report(testY, preds, target_names=le.classes_))

    # compute the raw accuracy with extra precision
    acc = accuracy_score(testY, preds)
    print("[INFO] score: {:.2f}%\n".format(acc * 100))
