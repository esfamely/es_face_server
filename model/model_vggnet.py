import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class ModelVGGNet:
    """
    vggnet简化版
    es 2018-10-08
    """

    @staticmethod
    def conv(x, w, b, name, stride=1):
        """
        卷积层
        """
        with tf.variable_scope('conv'):
            tf.summary.histogram('weight', w)
            tf.summary.histogram('biases', b)
            conv = tf.nn.conv2d(x,
                                filter=w,
                                strides=[1, stride, stride, 1],
                                padding='SAME',
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
                               name=name + '/beta', trainable=True, dtype=x.dtype)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out], dtype=x.dtype),
                                name=name + '/gamma', trainable=True, dtype=x.dtype)

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
    def pooling(x, k=2):
        """
        池化层
        """
        with tf.variable_scope('pooling'):
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    @staticmethod
    def fully_connected(x, w, b):
        """
        全连接层
        """
        with tf.variable_scope('fc'):
            tf.summary.histogram('weight', w)
            tf.summary.histogram('biases', b)
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
    def build(x, width, height, channels, classes, dropout=1.0, is_training=False):
        """
        创建网络结构
        """
        with tf.variable_scope('vggnet'):
            t = ModelVGGNet

            # 训练时，卷积层的dropout减半
            dropout_conv = tf.where(tf.less(dropout, 1.0), tf.div(dropout, 2), dropout)

            # 下降采样次数
            down_num = 2
            # 卷积层最后输出的宽度与高度
            w, h = int(width / (down_num * 2)), int(height / (down_num * 2))

            # 每层的规模
            c0, c1, c2, c3, c4 = channels, 16, 16, 32, 32
            f0, f1, f2 = w * h * c4, 256, classes

            # 初始化卷积层训练参数
            w_conv_1 = tf.get_variable('w_conv_1', [3, 3, c0, c1], initializer=t.w_initializer())
            b_conv_1 = tf.get_variable('b_conv_1', [c1], initializer=t.b_initializer())
            w_conv_2 = tf.get_variable('w_conv_2', [3, 3, c1, c2], initializer=t.w_initializer())
            b_conv_2 = tf.get_variable('b_conv_2', [c2], initializer=t.b_initializer())
            w_conv_3 = tf.get_variable('w_conv_3', [3, 3, c2, c3], initializer=t.w_initializer())
            b_conv_3 = tf.get_variable('b_conv_3', [c3], initializer=t.b_initializer())
            w_conv_4 = tf.get_variable('w_conv_4', [3, 3, c3, c4], initializer=t.w_initializer())
            b_conv_4 = tf.get_variable('b_conv_4', [c4], initializer=t.b_initializer())

            # 初始化全连接层训练参数
            w_fc_1 = tf.get_variable('w_fc_1', [f0, f1], initializer=t.w_initializer())
            b_fc_1 = tf.get_variable('b_fc_1', [f1], initializer=t.b_initializer())
            w_fc_2 = tf.get_variable('w_fc_2', [f1, f2], initializer=t.w_initializer())
            b_fc_2 = tf.get_variable('b_fc_2', [f2], initializer=t.b_initializer())

            # 输入
            x = tf.reshape(x, shape=[-1, width, height, channels])

            # 第一组网络结构：CONV => RELU => CONV => RELU => POOL
            x = t.conv(x, w_conv_1, b_conv_1, "conv1")
            x = tf.nn.elu(x)
            x = t.batch_norm(x, is_training, "batch_norm_c1")
            x = t.conv(x, w_conv_2, b_conv_2, "conv2")
            x = tf.nn.elu(x)
            x = t.batch_norm(x, is_training, "batch_norm_c2")
            x = t.pooling(x)
            #x = tf.nn.dropout(x, dropout_conv)

            # 第二组网络结构：CONV => RELU => CONV => RELU => POOL
            x = t.conv(x, w_conv_3, b_conv_3, "conv3")
            x = tf.nn.elu(x)
            x = t.batch_norm(x, is_training, "batch_norm_c3")
            x = t.conv(x, w_conv_4, b_conv_4, "conv4")
            x = tf.nn.elu(x)
            x = t.batch_norm(x, is_training, "batch_norm_c4")
            x = t.pooling(x)
            #x = tf.nn.dropout(x, dropout_conv)

            # 全连接层
            x = tf.reshape(x, [-1, w_fc_1.get_shape().as_list()[0]])
            x = t.fully_connected(x, w_fc_1, b_fc_1)
            x = tf.nn.elu(x)
            #x = t.batch_norm(x, is_training, "batch_norm_f1")
            x = tf.nn.dropout(x, dropout)
            x = t.fully_connected(x, w_fc_2, b_fc_2)

            # 输出
            logits = x
            prediction = tf.nn.softmax(logits)

            return logits, prediction
