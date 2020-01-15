import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class ModelResnet:
    """
    resnet简化版
    es 2018-10-12
    """

    @staticmethod
    def conv(x, cX, cY, c1, c2, id, stride=1, padding="SAME"):
        """
        卷积层，本版resnet不需要bias训练参数
        :param cX: 卷积核横向尺寸
        :param cY: 卷积核纵向尺寸
        :param c1: 上一层特征图(feature map)数目
        :param c2: 本层特征图(feature map)数目
        :param id: 卷积模块id，参与变量名的组成
        """
        with tf.variable_scope("conv", regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
            t = ModelResnet
            id = str(id)

            # 初始化卷积层训练参数
            w_conv = tf.get_variable("w_conv" + id, [cX, cY, c1, c2], initializer=t.w_initializer())

            tf.summary.histogram("weight", w_conv)
            conv = tf.nn.conv2d(x,
                                filter=w_conv,
                                strides=[1, stride, stride, 1],
                                padding=padding,
                                name="conv" + id)
            return conv

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
    def fully_connected(x, f1, f2, id):
        """
        全连接层
        :param f1: 上一层单元数
        :param f2: 本层单元数
        :param id: 全连接层id，参与变量名的组成
        """
        with tf.variable_scope("fc", regularizer=tf.contrib.layers.l2_regularizer(0.0001)):
            t = ModelResnet
            id = str(id)

            # 初始化全连接层训练参数
            w_fc = tf.get_variable('w_fc' + id, [f1, f2], initializer=t.w_initializer())
            b_fc = tf.get_variable('b_fc' + id, [f2], initializer=t.b_initializer())

            tf.summary.histogram("weight", w_fc)
            tf.summary.histogram("biases", b_fc)
            fc = tf.add(tf.matmul(x, w_fc), b_fc)

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
    def residual_module(x, c1, c2, stride, is_training, id, reduce=False):
        """
        resnet特有的结构模块，一条捷径与一系列cnn分道扬镳，最后会合
        :param c1: 上一层特征图(feature map)数目
        :param c2: 本层特征图(feature map)数目
        :param id: 卷积模块id，参与变量名的组成
        :param reduce: 是否以下降采样结束
        """
        t = ModelResnet

        # 捷径
        shortcut = x

        # 第1组cnn，特征图(feature map)数目只有预设的1/4
        x = t.batch_norm(x, is_training, "batch_norm" + str(id))
        act = tf.nn.elu(x)
        x = t.conv(act, 1, 1, c1, int(c2 * 0.25), id)
        id += 1

        # 第2组cnn，特征图(feature map)数目只有预设的1/4
        x = t.batch_norm(x, is_training, "batch_norm" + str(id))
        x = tf.nn.elu(x)
        x = t.conv(x, 3, 3, int(c2 * 0.25), int(c2 * 0.25), id, stride=stride)
        id += 1

        # 第3组cnn
        x = t.batch_norm(x, is_training, "batch_norm" + str(id))
        x = tf.nn.elu(x)
        x = t.conv(x, 1, 1, int(c2 * 0.25), c2, id)
        id += 1

        # 如果以下降采样结束，捷径需单独处理，cnn群组已经在第2组实现stride下降
        if reduce:
            # 卷积前需先经 BN => ACT 处理，所以取第1组cnn中的act作为输入
            shortcut = t.conv(act, 1, 1, c1, c2, id, stride=stride)
            id += 1

        # 会合
        x = tf.add_n([x, shortcut])

        return x

    @staticmethod
    def build(x, width, height, channels, classes, dropout=1.0, is_training=False):
        """
        创建网络结构
        """
        with tf.variable_scope("resnet"):
            t = ModelResnet

            # 输入
            x = tf.reshape(x, shape=[-1, width, height, channels])

            # 定义网络核心结构与规模
            stages = (3, 3, 3)
            filters = (16, 16, 32, 64)

            # 第一组网络结构
            id = 1
            x = t.batch_norm(x, is_training, "batch_norm" + str(id))
            x = t.conv(x, 3, 3, channels, filters[0], id)
            id += 1
            c_pre = filters[0]

            # 按定义创建网络结构
            for i in range(0, len(stages)):
                #print("i: {}".format(i))
                # 除了第一组，每组都以下降采样(stride=2)作开头
                stride = 1 if i == 0 else 2
                x = t.residual_module(x, c_pre, filters[i + 1], stride, is_training, id,
                                      reduce=True)
                # residual_module里面id已经增加不止一次
                id += 4
                c_pre = filters[i + 1]

                for j in range(0, stages[i] - 1):
                    #print("j: {}".format(j))
                    x = t.residual_module(x, c_pre, filters[i + 1], 1, is_training, id)
                    id += 3
                    c_pre = filters[i + 1]

            # 再一组网络结构：BN => ACT => POOL
            x = t.batch_norm(x, is_training, "batch_norm" + str(id))
            x = tf.nn.elu(x)
            x = tf.nn.avg_pool(x,
                               ksize=[1, 8, 8, 1],
                               strides=[1, 8, 8, 1],
                               padding="VALID")

            # 全连接层
            f1 = 1 * 1 * c_pre
            x = tf.reshape(x, [-1, f1])
            x = tf.nn.dropout(x, dropout)
            x = t.fully_connected(x, f1, classes, 1)

            # 输出
            logits = x
            prediction = tf.nn.softmax(logits)

            return logits, prediction
