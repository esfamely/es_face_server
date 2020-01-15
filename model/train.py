import tensorflow as tf
import datetime as dt
import os
from pathlib import Path


class Train:
    """
    model训练
    es 2018-10-08
    """

    @staticmethod
    def save_model(sess, saver, epoch, best_acc_vat, name):
        """
        保存最佳的model
        """
        # 保存路径
        model_dir = "../output/ckpt_" + name + "/"
        model_path = model_dir + name + "_{:02d}_{:.2f}.ckpt"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 先清空旧记录，只保留最佳的
        lFile = list(Path(model_dir).glob("*.*"))
        for file in lFile:
            #print(file)
            os.remove(file.absolute())

        # 保存
        save_path = saver.save(sess, model_path.format(epoch + 1, best_acc_vat * 100))
        print("[INFO] save best model: %s" % save_path)

    @staticmethod
    def train(model, trainX, vatX, trainY, vatY, name, hp=None):
        """
        训练
        """
        t = Train

        # 训练信息
        if hp is None:
            learning_rate = 0.01
            max_epochs = 100
            batch_size = 32
        else:
            learning_rate = hp["learning_rate"]
            max_epochs = hp["max_epochs"]
            batch_size = hp["batch_size"]

        graphs_dir = "../output/graphs_" + name

        # 提取各维度信息
        width = trainX.shape[1]
        height = trainX.shape[2]
        channels = trainX.shape[3]
        classes = trainY.shape[1]

        with tf.variable_scope('placeholder'):
            X = tf.placeholder(tf.float32, [None, width, height, channels])
            Y = tf.placeholder(tf.float32, [None, classes])
            Dropout = tf.placeholder(tf.float32)
            Is_training = tf.placeholder("bool")

        with tf.variable_scope('prediction'):
            logits, prediction = model.build(X, width, height, channels, classes,
                                                       dropout=Dropout,
                                                       is_training=Is_training)

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
            regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss += regularization_loss
            tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            # momentum + nesterov
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            train_op = optimizer.minimize(loss)

            # 使用可变的learning_rate
            '''decay_steps = 3
            decay_rate = 0.95
            global_step = tf.Variable(0, trainable=False)
            # 指数衰减：decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
            learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=decay_steps,
                                                       decay_rate=decay_rate,
                                                       staircase=True)
            optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
            train_op = optimizer.minimize(loss, global_step=global_step)'''

        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

        init = tf.global_variables_initializer()

        # 保存最佳的model
        saver = tf.train.Saver()

        # 训练开始
        with tf.Session() as sess:
            sess.run(init)

            # 记录验证集上最佳的准确率
            best_acc_vat = 0

            # 记录监控图表
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(graphs_dir)
            writer.add_graph(sess.graph)

            for epoch in range(max_epochs):
                start_time = dt.datetime.now()

                index = 0
                train_size = len(trainX)
                while index < train_size:
                    index_end = index + batch_size
                    if index_end > train_size:
                        index_end = train_size
                    _, l = sess.run([train_op, loss],
                                    feed_dict={X: trainX[index:index_end],
                                               Y: trainY[index:index_end],
                                               Dropout: 0.5,
                                               Is_training: True})
                    index += batch_size
                    print("[INFO] epoch: {:02d}, step: {}/{}, loss: {:.6f}".
                          format(epoch + 1, index, train_size, l))

                # 本期损失、准确率
                loss_train, acc_train = sess.run([loss, accuracy],
                                                 feed_dict={X: trainX,
                                                            Y: trainY,
                                                            Dropout: 1.0,
                                                            Is_training: False})
                loss_vat, acc_vat, summary = sess.run([loss, accuracy, merged_summary],
                                                      feed_dict={X: vatX,
                                                                 Y: vatY,
                                                                 Dropout: 1.0,
                                                                 Is_training: False})
                str_loss = "loss_train: {:.6f}, loss_vat: {:.6f}, ".format(loss_train, loss_vat)
                str_acc = "acc_train: {:.4f}, acc_vat: {:.4f}, ".format(acc_train, acc_vat)

                # 记录验证集上最佳的准确率
                if (acc_vat > best_acc_vat):
                    best_acc_vat = acc_vat

                    # 保存最佳的model
                    t.save_model(sess, saver, epoch, best_acc_vat, name)

                # 记录监控图表
                writer.add_summary(summary, epoch)

                # 本期耗时
                end_time = dt.datetime.now()
                times = (end_time - start_time).seconds
                str_times = "times: {:02d}:{:02d}".format(int(times / 60), times % 60)

                # 显示本期训练反馈
                print("[INFO] epoch: {:02d}, ".format(epoch + 1) + str_loss + str_acc + str_times)
