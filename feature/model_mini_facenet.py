import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class MiniFacenet:
    """
    Facenet简化版
    es 2018-11-08
    """

    def __init__(self, channels):
        # 定义网络核心结构与规模
        self.css = [
            [channels, 32, 32],
            [32, 64, 64],
            [64, 128, 128]
        ]
        # 全连接层单元数
        self.f0 = 5 * 5 * self.css[-1][-1]
        self.f1 = 128

        # 正则化
        self.regularizer = tf.contrib.layers.l2_regularizer(0.0001)

        # 初始化训练参数
        self.init_weights_biases()

    def init_weights_biases(self):
        """
        初始化训练参数
        """
        t = MiniFacenet

        # 初始化卷积层训练参数
        self.weights = {}
        self.biases = {}
        for i, cs in enumerate(self.css):
            num = len(cs) - 1
            ci = 0

            # 多个cnn重叠
            for j in range(num):
                sn = "{}_{}".format(i, j)
                w_name = "w_conv_" + sn
                b_name = "b_conv_" + sn

                with tf.variable_scope("conv_" + sn, regularizer=self.regularizer):
                    w_conv = tf.get_variable(w_name, [3, 3, cs[ci], cs[ci + 1]],
                                             initializer=t.w_initializer())
                    b_conv = tf.get_variable(b_name, [cs[ci + 1]],
                                             initializer=t.b_initializer())
                    ci += 1

                    self.weights[w_name] = w_conv
                    self.biases[b_name] = b_conv

                    tf.summary.histogram("weight", w_conv)
                    tf.summary.histogram("biases", w_conv)

        # 初始化全连接层训练参数
        with tf.variable_scope("fc_1", regularizer=self.regularizer):
            w_fc_1 = tf.get_variable("w_fc_1", [self.f0, self.f1], initializer=t.w_initializer())
            b_fc_1 = tf.get_variable("b_fc_1", [self.f1], initializer=t.b_initializer())

            self.weights["w_fc_1"] = w_fc_1
            self.biases["b_fc_1"] = b_fc_1

            tf.summary.histogram("weight", w_fc_1)
            tf.summary.histogram("biases", b_fc_1)

    @staticmethod
    def conv(x, w, b, name, stride=1):
        """
        卷积层
        """
        conv = tf.nn.conv2d(x,
                            filter=w,
                            strides=[1, stride, stride, 1],
                            padding="SAME",
                            name=name)
        return tf.nn.bias_add(conv, b)

    @staticmethod
    def batch_norm(x, is_training, name):
        """
        batch normalization
        """
        is_training = tf.convert_to_tensor(is_training, dtype=tf.bool)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = control_flow_ops.cond(is_training,
                                          mean_var_with_update,
                                          lambda: (ema.average(batch_mean), ema.average(batch_var)))
        bn = tf.nn.batch_normalization(x, mean, var, 0.0, 1.0, 1e-3)

        return bn

    @staticmethod
    def pooling(x, k=2):
        """
        池化层
        """
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

    @staticmethod
    def fully_connected(x, w, b):
        """
        全连接层
        """
        fc = tf.add(tf.matmul(x, w), b)
        return fc

    @staticmethod
    def w_initializer():
        """
        weights初始化
        """
        return tf.random_normal_initializer(stddev=0.1)

    @staticmethod
    def b_initializer():
        """
        bias初始化
        """
        return tf.constant_initializer()

    def vgg_module(self, x, num, cs, id, is_training):
        """
        vggnet特有的结构模块
        """
        t = MiniFacenet
        id = str(id)
        ci = 0

        # 多个cnn重叠
        for i in range(num):
            # 初始化卷积层训练参数
            sn = "{}_{}".format(id, i)
            w_name = "w_conv_" + sn
            b_name = "b_conv_" + sn

            x = t.conv(x, self.weights[w_name], self.biases[b_name], "conv_" + sn)
            x = tf.nn.relu(x)
            x = t.batch_norm(x, is_training, "bn_conv_" + sn)

        x = t.pooling(x)

        return x

    def conv_net(self, x, width, height, channels, dropout, is_training):
        t = MiniFacenet

        # 输入
        x = tf.reshape(x, shape=[-1, width, height, channels])

        # 按定义创建网络结构
        for i, cs in enumerate(self.css):
            num = len(cs) - 1
            x = self.vgg_module(x, num, cs, i, is_training)

        # 将全部特征图(feature map)展开，并使用dropout
        x = tf.reshape(x, [-1, self.f0])
        x = tf.nn.dropout(x, dropout)

        # 全连接层
        x = t.fully_connected(x, self.weights["w_fc_1"], self.biases["b_fc_1"])

        return x

    def build(self, x1, x2, x3, width, height, channels, dropout=1.0, is_training=False):
        out1 = self.conv_net(x1, width, height, channels, dropout, is_training)
        out2 = self.conv_net(x2, width, height, channels, dropout, is_training)
        out3 = self.conv_net(x3, width, height, channels, dropout, is_training)

        return out1, out2, out3

    @staticmethod
    def triplet_loss(f1, f2, f3):
        a = tf.constant(1, dtype=tf.float32)
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(f1, f2)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(f1, f3)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), a)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        return loss
