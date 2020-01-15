import tensorflow as tf


class Prediction:
    """
    model分类预测
    es 2018-10-10
    """

    def __init__(self, model, width, height, channels, classes, model_path):
        self.load_model(model, width, height, channels, classes, model_path)

    def load_model(self, model, width, height, channels, classes, model_path):
        """
        加载最佳的model
        """
        with tf.variable_scope('placeholder'):
            X = tf.placeholder(tf.float32, [None, width, height, channels])
            Y = tf.placeholder(tf.float32, [None, classes])
            Dropout = tf.placeholder(tf.float32)
            Is_training = tf.placeholder("bool")

        with tf.variable_scope('prediction'):
            logits, prediction = model.build(X, width, height, channels, classes,
                                                       dropout=Dropout,
                                                       is_training=Is_training)

        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        # 保存最佳的model
        saver = tf.train.Saver()

        # 加载开始
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载最佳的model
        saver.restore(sess, model_path)
        print("[INFO] load best model: %s" % model_path)

        self.sess = sess
        self.graphs = {
            "X": X,
            "Y": Y,
            "Dropout": Dropout,
            "Is_training": Is_training,
            "logits": logits,
            "prediction": prediction,
            "accuracy": accuracy
        }

    def check_model(self, vatX, vatY):
        """
        确认model是否成功加载
        """
        sess = self.sess
        graphs = self.graphs

        # 准确率确认
        acc_vat = sess.run(graphs["accuracy"], feed_dict={graphs["X"]: vatX,
                                                          graphs["Y"]: vatY,
                                                          graphs["Dropout"]: 1.0,
                                                          graphs["Is_training"]: False})
        print("[INFO] best model ok: {:.6f}".format(acc_vat))
        return acc_vat > 0.5

    def prediction(self, sample):
        """
        分类预测
        """
        sess = self.sess
        graphs = self.graphs

        # 计算属于每个分类的概率
        prediction = sess.run(graphs["prediction"],
                              feed_dict={graphs["X"]: sample,
                                         graphs["Dropout"]: 1.0,
                                         graphs["Is_training"]: False})

        return prediction
