import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class ModelInception:
    """
    inception(GoogLeNet)简化版
    es 2018-10-11
    """

    @staticmethod
    def conv(x, w, b, name, stride=1, padding="SAME"):
        """
        卷积层
        """
        with tf.variable_scope("conv"):
            tf.summary.histogram("weight", w)
            tf.summary.histogram("biases", b)
            conv = tf.nn.conv2d(x,
                                filter=w,
                                strides=[1, stride, stride, 1],
                                padding=padding,
                                name=name)
            return tf.nn.bias_add(conv, b)

    @staticmethod
    def batch_norm(x, is_training, name):
        #bn = tf.contrib.layers.batch_norm(x)

        # 以下的计算方法可提高速度与准确率
        with tf.variable_scope(name):
            is_training = tf.convert_to_tensor(is_training, dtype=tf.bool)
            n_out = int(x.get_shape()[-1])
            beta = tf.Variable(tf.constant(0.0, shape=[n_out], dtype=x.dtype),
                               name=name + "/beta", trainable=True, dtype=x.dtype)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                                name=name + "/gamma", trainable=True, dtype=x.dtype)

            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            ema = tf.train.ExponentialMovingAverage(decay=0.9)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = control_flow_ops.cond(is_training,
                                              mean_var_with_update,
                                              lambda: (ema.average(batch_mean),
                                                       ema.average(batch_var)))
            bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

        return bn

    @staticmethod
    def pooling(x, k=3, stride=2):
        """
        池化层
        """
        with tf.variable_scope("pooling"):
            return tf.nn.max_pool(x,
                                  ksize=[1, k, k, 1],
                                  strides=[1, stride, stride, 1],
                                  padding="VALID")

    @staticmethod
    def fully_connected(x, w, b):
        """
        全连接层
        """
        with tf.variable_scope("fc"):
            tf.summary.histogram("weight", w)
            tf.summary.histogram("biases", b)
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

    @staticmethod
    def conv_module(x, c1, c2, kX, kY, stride, is_training, id, padding="SAME"):
        """
        卷积模块
        :param c1: 上一层特征图(feature map)数目
        :param c2: 本层特征图(feature map)数目
        :param kX: 卷积核横向尺寸
        :param kY: 卷积核纵向尺寸
        :param id: 卷积模块id，参与变量名的组成
        """
        t = ModelInception

        id = str(id)

        # 初始化卷积层训练参数
        w_conv = tf.get_variable("w_conv_" + id, [kX, kY, c1, c2], initializer=t.w_initializer())
        b_conv = tf.get_variable("b_conv_" + id, [c2], initializer=t.b_initializer())

        # 定义模式：CONV => BN => RELU
        x = t.conv(x, w_conv, b_conv, "conv" + id, stride=stride, padding=padding)
        x = t.batch_norm(x, is_training, "batch_norm_c" + id)
        x = tf.nn.elu(x)

        return x

    @staticmethod
    def inception_module(x, c_pre, c1x1, c3x3, is_training, c1x1_id, c3x3_id):
        """
        inception特有的结构模块，若干个不同的卷积结构并联起来
        :param c_pre: 上一层特征图(feature map)数目
        :param c1x1: 1x1卷积核，本层特征图(feature map)数目
        :param c3x3: 3x3卷积核，本层特征图(feature map)数目
        :param c1x1_id: 1x1卷积模块id，参与变量名的组成
        :param c3x3_id: 3x3卷积模块id，参与变量名的组成
        """
        t = ModelInception

        # 若干个不同的卷积结构
        conv_1x1 = t.conv_module(x, c_pre, c1x1, 1, 1, 1, is_training, c1x1_id)
        conv_3x3 = t.conv_module(x, c_pre, c3x3, 3, 3, 1, is_training, c3x3_id)

        # 并联起来
        x = tf.concat([conv_1x1, conv_3x3], axis=3)

        return x

    @staticmethod
    def downsample_module(x, c1, c2, is_training, id):
        """
        下降采样模块
        """
        t = ModelInception

        # 大间隔卷积下降采样
        conv_3x3 = t.conv_module(x, c1, c2, 3, 3, 2, is_training, id, padding="VALID")
        # pooling下降采样
        pool = t.pooling(x)

        # 并联起来
        x = tf.concat([conv_3x3, pool], axis=3)

        return x

    @staticmethod
    def build(x, width, height, channels, classes, dropout=1.0, is_training=False):
        """
        创建网络结构
        """
        with tf.variable_scope("inception"):
            t = ModelInception

            # 输入
            x = tf.reshape(x, shape=[-1, width, height, channels])

            # 卷积模块开始
            c0 = 96
            id = 1
            x = t.conv_module(x, channels, c0, 3, 3, 1, is_training, id)
            id += 1

            # 定义网络核心结构与规模
            c = [
                [16, 16],
                [16, 24],
                [40],
                [56, 24],
                [48, 32],
                [40, 40],
                [24, 48],
                [48],
                [88, 80],
                [88, 80]
            ]
            # 上一层特征图(feature map)数目
            c_pre = c0
            # 按定义创建网络结构
            for i, cc in enumerate(c):
                #print(i)
                if len(cc) > 1:
                    c1x1, c3x3 = cc[0], cc[1]
                    c1x1_id = id
                    id += 1
                    c3x3_id = id
                    id += 1
                    x = t.inception_module(x, c_pre, c1x1, c3x3, is_training, c1x1_id, c3x3_id)
                    c_pre = cc[0] + cc[1]
                else:
                    c1, c2 = c_pre, cc[0]
                    x = t.downsample_module(x, c1, c2, is_training, id)
                    c_pre = cc[0] + c_pre
                    id += 1

            # 大幅度下降采样
            x = tf.nn.avg_pool(x,
                               ksize=[1, 7, 7, 1],
                               strides=[1, 7, 7, 1],
                               padding="VALID")

            # 将全部特征图(feature map)展开，并使用dropout
            f0 = 1 * 1 * c_pre
            x = tf.reshape(x, [-1, f0])
            x = tf.nn.dropout(x, dropout)

            # 初始化全连接层训练参数
            w_fc_1 = tf.get_variable('w_fc_1', [f0, classes], initializer=t.w_initializer())
            b_fc_1 = tf.get_variable('b_fc_1', [classes], initializer=t.b_initializer())

            # 全连接层
            x = t.fully_connected(x, w_fc_1, b_fc_1)

            # 输出
            logits = x
            prediction = tf.nn.softmax(logits)

            return logits, prediction
