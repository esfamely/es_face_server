import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from preprocessing.simple_preprocessor import SimplePreprocessor
from preprocessing.hist_preprocessor import HistPreprocessor
from preprocessing.image2array_preprocessor import ImageToArrayPreprocessor
from dataset.simple_dataset_loader import SimpleDatasetLoader
from model.model_vggnet import ModelVGGNet
from model.model_inception import ModelInception
from model.model_resnet import ModelResnet
from model.train import Train
from model.prediction import Prediction

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
image_dir = "D:/s5/cv_python/deep_learning/dataset/easyface"

# initialize the image preprocessors
w = 32
sp = SimplePreprocessor(w, w)
hist = HistPreprocessor(gray=True)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, hist, iap])
(data, labels, le) = sdl.load(image_dir, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits
(trainX, vatX, trainY, vatY) = train_test_split(data, labels, test_size=0.5, random_state=10)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
vatY = LabelBinarizer().fit_transform(vatY)

# train
Train.train(ModelVGGNet, trainX, vatX, trainY, vatY, "mini_vggnet")
#Train.train(ModelInception, trainX, vatX, trainY, vatY, "mini_inception")
'''Train.train(ModelResnet, trainX, vatX, trainY, vatY, "resnet", {
    "learning_rate": 0.1,
    "max_epochs": 100,
    "batch_size": 32
})'''

# prediction
model_path = "D:/s5/cv_python/es_face/output/ckpt_resnet/ok/resnet_88_97.78.ckpt"
prediction = Prediction(ModelResnet, w, w, 1, 5, model_path)
r = prediction.check_model(vatX, vatY)
if r:
    num_acc = 0
    num_ok = 0
    num_ok_acc = 0
    num_sample = vatY.shape[0]
    np.set_printoptions(suppress=True)
    for i, sample in enumerate(vatX):
        sample_ = np.expand_dims(sample, axis=0)
        pred = prediction.prediction(sample_)

        pred_max = np.max(pred)
        # print("----- ----- ----- ----- ----- {}".format(np.around(pred_max, decimals=6)))
        if pred_max > 0.9:
            num_ok += 1
            if np.equal(np.argmax(pred), np.argmax(vatY[i])):
                num_ok_acc += 1

        if np.equal(np.argmax(pred), np.argmax(vatY[i])):
            num_acc += 1
        else:
            print("{} | {}".format(np.around(pred, decimals=6), vatY[i]))

    print(num_acc / num_sample)
    print(num_ok / num_sample)
    print(num_ok_acc / num_ok)
