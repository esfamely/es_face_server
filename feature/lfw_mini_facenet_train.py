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

imgs_anchor, imgs_positive, imgs_negative = LfwReadImages.make_triplet(lfw_path, pairs_train_path)
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
print("img {} ok !".format(len(img_s1)))

learning_rate = 0.01
max_epochs = 100
batch_size = 16

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

# Start training
with tf.Session() as sess:
    sess.run(init)

    # 记录监控图表
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(graphs_dir)
    writer.add_graph(sess.graph)

    for epoch in range(max_epochs):
        start_time = dt.datetime.now()
        gl = 0
        index = 0
        ts = 0
        train_size = len(img_s1)
        while index < train_size:
            index_end = index + batch_size
            if index_end > train_size:
                index_end = train_size
            _, l = sess.run([train_op, loss], feed_dict={X1: img_s1[index:index_end],
                                                         X2: img_s2[index:index_end],
                                                         X3: img_s3[index:index_end],
                                                         Dropout: 0.5,
                                                         Is_Training: True})
            gl += l
            ts += 1
            index += batch_size
            print("epoch: {:02d}, step: {}/{}, loss: {:.6f}".
                  format(epoch + 1, index, train_size, l))

        # 记录监控图表
        bl, l, summary = sess.run([base_loss, loss, merged_summary], feed_dict={X1: img_s1,
                                                                                X2: img_s2,
                                                                                X3: img_s3,
                                                                                Dropout: 0.5,
                                                                                Is_Training: True})
        writer.add_summary(summary, epoch)

        gl_mean = gl / ts
        end_time = dt.datetime.now()
        times = (end_time - start_time).seconds
        times = "{:02d}:{:02d}".format(int(times / 60), times % 60)
        print("----- ----- epoch: {:02d}, base_loss: {:.6f}, loss: {:.6f}, times: {}".format(
            epoch + 1, bl, l, times))

        # Save model weights to disk
        if bl < best_loss:
            save_path = saver.save(sess, model_path.format(epoch + 1, l))
            print("Model saved in file: %s" % save_path)
            best_loss = bl
