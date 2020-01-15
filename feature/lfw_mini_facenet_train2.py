import cv2
import tensorflow as tf
from feature.lfw_read_images import LfwReadImages
from feature.model_mini_facenet import MiniFacenet
import datetime as dt

root_dir = "D:/s5/dataset/"
lfw_path = root_dir + "lfw_160/"
pairs_train_path = root_dir + "lfw_txt/pairsDevTrain.txt"

width = 40
height = 40
channels = 3

learning_rate = 0.01
max_epochs = 300
batch_size = 32

with tf.variable_scope('placeholder'):
    X1 = tf.placeholder("float", [None, width, height, channels])
    X2 = tf.placeholder("float", [None, width, height, channels])
    X3 = tf.placeholder("float", [None, width, height, channels])
    Dropout = tf.placeholder(tf.float32)
    Is_Training = tf.placeholder("bool")

with tf.variable_scope('loss'):
    mini_facenet = MiniFacenet(channels)
    out1, out2, out3 = mini_facenet.build(X1, X2, X3, width, height, channels,
                                          dropout=Dropout, is_training=Is_Training)

    base_loss = MiniFacenet.triplet_loss(out1, out2, out3)
    # 正则化损失
    regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = base_loss + regularization_loss

    tf.summary.scalar('base_loss', base_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    train_op = optimizer.minimize(base_loss)

init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
model_path = root_dir + "lfw_txt/facenet/160_40_1_{:02d}_{:.2f}.ckpt"
graphs_dir = root_dir + "lfw_txt/facenet/graphs"
best_loss = 1.0

## 每期都随机抽取不同部分样本进行训练
imgs_anchor, imgs_positive, imgs_negative = LfwReadImages.make_triplet2(
    lfw_path, pairs_train_path, cell_size=3)
img_s1, img_s2, img_s3 = [], [], []
for img1, img2, img3 in zip(imgs_anchor, imgs_positive, imgs_negative):
    img1_ = cv2.imread(img1)
    img2_ = cv2.imread(img2)
    img3_ = cv2.imread(img3)
    img1_ = cv2.resize(img1_, (width, height))
    img2_ = cv2.resize(img2_, (width, height))
    img3_ = cv2.resize(img3_, (width, height))
    img_s1.append(img1_ / 255)
    img_s2.append(img2_ / 255)
    img_s3.append(img3_ / 255)
print("imgs size: {}".format(len(imgs_anchor)))

# Start training
with tf.Session() as sess:
    sess.run(init)

    # 记录监控图表
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graphs_dir)
    writer.add_graph(sess.graph)

    for epoch in range(max_epochs):
        start_time = dt.datetime.now()

        # 分批训练
        gl = 0
        ts = 0
        index = 0
        train_size = len(imgs_anchor)
        while index < train_size:
            index_end = index + batch_size
            if index_end > train_size:
                index_end = train_size

            _, bl = sess.run([train_op, base_loss], feed_dict={X1: img_s1[index : index_end],
                                                               X2: img_s2[index : index_end],
                                                               X3: img_s3[index : index_end],
                                                               Dropout: 0.5,
                                                               Is_Training: True})
            gl += bl
            ts += 1

            index += batch_size
            print("epoch: {:02d}, step: {}/{}, base_loss: {:.6f}".
                  format(epoch + 1, index_end, train_size, bl))

        # 记录监控图表
        bl, rl, summary = sess.run([base_loss, regularization_loss, merged_summary],
                                   feed_dict={X1: img_s1[0 : batch_size],
                                              X2: img_s2[0 : batch_size],
                                              X3: img_s3[0 : batch_size],
                                              Dropout: 0.5,
                                              Is_Training: True})
        writer.add_summary(summary, epoch)

        # Save model weights to disk
        gl_mean = gl / ts
        if gl_mean < best_loss:
            save_path = saver.save(sess, model_path.format(epoch + 1, gl_mean))
            print("Model saved in file: %s" % save_path)
            best_loss = gl_mean

        # 释放资源
        '''img_s1.clear()
        img_s2.clear()
        img_s3.clear()
        imgs_anchor.clear()
        imgs_positive.clear()
        imgs_negative.clear()'''

        # 每期结果
        end_time = dt.datetime.now()
        times = (end_time - start_time).seconds
        times = "{:02d}:{:02d}".format(int(times / 60), times % 60)
        info = "----- ----- epoch: {:02d}, base_loss: {:.6f}, regularization_loss: {:.6f}"
        info += ", gl_mean: {}, times: {}"
        print(info.format(epoch + 1, bl, rl, gl_mean, times))
