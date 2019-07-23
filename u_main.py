# -- encoding:utf-8 --
"""
Created by MengCheng Ren on 2019/5/26
"""
import tensorflow as tf
import argparse
import numpy as np
import cv2
import os
import tensorlayer as tl
import tensorflow.contrib.slim as slim
from ops import acc_compute, curve_save, image_save


class Tensors:
    def __init__(self, config):
        with tf.device('/gpu:0'):
            self.lr = config.lr

            self.x_input = tf.placeholder(tf.float32, [None, 512, 512, 1], name='input')
            self.y_label = tf.placeholder(tf.float32, [None, 512, 512, 2], name='label')
            self.keep_prob = tf.placeholder(tf.float32, None, name='keep_prob')

            self.y_output = self.u_net(self.x_input, self.keep_prob)

            target = tf.reshape(self.y_label, [-1, 2])
            logistic = tf.reshape(self.y_output, [-1, 2])

            self.loss = config.beta * tl.cost.dice_coe(self.y_output, self.y_label) + (1-config.beta) * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=1 - logistic))

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(self.lr, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = optimizer.minimize(self.loss, global_step=global_step)

            self.acc = acc_compute(self.y_output, 1-self.y_label)

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('acc', self.acc)
            tf.summary.scalar('lr', learning_rate)
            self.summary = tf.summary.merge_all()

    def u_net(self, x, keep_prob):
        filters = {
            'w1': tf.get_variable('W0', shape=(3, 3, 1, 32),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w2': tf.get_variable('W1', shape=(3, 3, 32, 32),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w3': tf.get_variable('W2', shape=(3, 3, 32, 16),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w4': tf.get_variable('W3', shape=(3, 3, 16, 16),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w5': tf.get_variable('W4', shape=(3, 3, 16, 32),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w6': tf.get_variable('W5', shape=(3, 3, 32, 32),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w7': tf.get_variable('W6', shape=(3, 3, 32, 64),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w8': tf.get_variable('W7', shape=(3, 3, 64, 64),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w9': tf.get_variable('W8', shape=(3, 3, 64, 128),
                                  initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w10': tf.get_variable('W9', shape=(3, 3, 128, 128),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w11': tf.get_variable('W10', shape=(3, 3, 192, 64),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w12': tf.get_variable('W11', shape=(3, 3, 64, 64),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w13': tf.get_variable('W12', shape=(3, 3, 96, 32),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w14': tf.get_variable('W13', shape=(3, 3, 32, 32),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w15': tf.get_variable('W14', shape=(3, 3, 48, 16),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w16': tf.get_variable('W15', shape=(3, 3, 16, 16),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w17': tf.get_variable('W16', shape=(3, 3, 16, 32),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w18': tf.get_variable('W17', shape=(3, 3, 32, 32),
                                   initializer=tf.contrib.layers.variance_scaling_initializer()),
            'w19': tf.get_variable('W18', shape=(3, 3, 32, 2),
                                   initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        biases = {
            'b1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b2': tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b3': tf.get_variable('B2', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b4': tf.get_variable('B3', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b5': tf.get_variable('B4', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b6': tf.get_variable('B5', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b7': tf.get_variable('B6', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b8': tf.get_variable('B7', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b9': tf.get_variable('B8', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b10': tf.get_variable('B9', shape=(128), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b11': tf.get_variable('B10', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b12': tf.get_variable('B11', shape=(64), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b13': tf.get_variable('B12', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b14': tf.get_variable('B13', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b15': tf.get_variable('B14', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b16': tf.get_variable('B15', shape=(16), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b17': tf.get_variable('B16', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b18': tf.get_variable('B17', shape=(32), initializer=tf.contrib.layers.variance_scaling_initializer()),
            'b19': tf.get_variable('B18', shape=(2), initializer=tf.contrib.layers.variance_scaling_initializer())
        }

        def conv2d(images, filter, bias, stride=(1, 1)):
            new_input = tf.nn.conv2d(images, filter, strides=[1, stride[0], stride[1], 1], padding='SAME')
            new_input = tf.nn.bias_add(new_input, bias)
            return tf.nn.relu(new_input)

        def conv(images, filter, bias, stride=(1, 1)):
            new_input = tf.nn.conv2d(images, filter, strides=[1, stride[0], stride[1], 1], padding='SAME')
            new_input = tf.nn.bias_add(new_input, bias)
            return new_input

        def maxpool2d(images, kernel_size=(2, 2)):
            return tf.nn.max_pool(images, ksize=[1, kernel_size[0], kernel_size[1], 1],
                                  strides=[1, kernel_size[0], kernel_size[1], 1], padding='VALID')

        def dropout(images, dropout_rate):
            return tf.nn.dropout(images, 1 - dropout_rate)

        def upsample(images, size=(2, 2)):
            new_height = size[0] * int(images.get_shape()[1])
            new_width = size[1] * int(images.get_shape()[2])

            return tf.image.resize_images(images, size=[new_height, new_width],
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        def batchnorm(net):
            return slim.batch_norm(net, decay=0.9, epsilon=0.001)

        with tf.variable_scope('u_net'):
            conv1 = conv2d(x, filters['w1'], biases['b1'])
            conv1 = dropout(conv1, keep_prob)
            conv1 = conv2d(conv1, filters['w2'], biases['b2'])
            up1 = upsample(conv1, (2, 2))
            #
            conv2 = conv2d(up1, filters['w3'], biases['b3'])
            conv2 = dropout(conv2, keep_prob)
            conv2 = conv2d(conv2, filters['w4'], biases['b4'])
            pool1 = maxpool2d(conv2, (2, 2))
            #
            conv3 = conv2d(pool1, filters['w5'], biases['b5'])
            conv3 = dropout(conv3, keep_prob)
            conv3 = conv2d(conv3, filters['w6'], biases['b6'])
            pool2 = maxpool2d(conv3, (2, 2))
            #
            conv4 = conv2d(pool2, filters['w7'], biases['b7'])
            conv4 = dropout(conv4, keep_prob)
            conv4 = conv2d(conv4, filters['w8'], biases['b8'])
            pool3 = maxpool2d(conv4, (2, 2))
            #
            conv5 = conv2d(pool3, filters['w9'], biases['b9'])
            conv5 = dropout(conv5, keep_prob)
            conv5 = conv2d(conv5, filters['w10'], biases['b10'])

            up2 = upsample(conv5, (2, 2))
            up2 = tf.concat([up2, conv4], 3)
            conv6 = conv2d(up2, filters['w11'], biases['b11'])
            conv6 = dropout(conv6, keep_prob)
            conv6 = conv2d(conv6, filters['w12'], biases['b12'])
            #
            up3 = upsample(conv6, (2, 2))
            up3 = tf.concat([up3, conv3], 3)
            conv7 = conv2d(up3, filters['w13'], biases['b13'])
            conv7 = dropout(conv7, keep_prob)
            conv7 = conv2d(conv7, filters['w14'], biases['b14'])
            #
            up4 = upsample(conv7, (2, 2))
            up4 = tf.concat([up4, conv2], 3)
            conv8 = conv2d(up4, filters['w15'], biases['b15'])
            conv8 = dropout(conv8, keep_prob)
            conv8 = conv2d(conv8, filters['w16'], biases['b16'])
            #
            pool4 = maxpool2d(conv8, (2, 2))
            conv9 = conv2d(pool4, filters['w17'], biases['b17'])
            conv9 = dropout(conv9, keep_prob)
            conv9 = conv2d(conv9, filters['w18'], biases['b18'])

            conv10 = conv(conv9, filters['w19'], biases['b19'])
            conv10 = batchnorm(conv10)
            conv10 = tf.nn.softmax(conv10)

        return conv10


class Unet:
    def __init__(self, config):
        self.config = config
        self.samples = Samples(self.config)
        graph = tf.Graph()
        with graph.as_default():
            self.tensor = Tensors(self.config)
            conf = tf.ConfigProto(allow_soft_placement=True)
            conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=conf, graph=graph)
            self.saver = tf.train.Saver()
            try:
                self.saver.restore(self.sess, self.config.model)
                print('Restore model success!')
            except:
                self.sess.run(tf.global_variables_initializer())
                print("Failed to restore model!")

    def train(self):
        file_writer = tf.summary.FileWriter(self.config.summary_path, self.sess.graph)
        steps = self.samples.num // self.config.batch_size
        loss_summary = []
        acc_summary = []
        x_datas, labels = self.samples.data()
        for epoch in range(self.config.epochs):
            loss_list = []
            acc_list = []
            for step in range(steps):
                x_data = x_datas[step*self.config.batch_size:(step + 1)*self.config.batch_size, :, :, :]
                label = labels[step*self.config.batch_size:(step + 1)*self.config.batch_size, :, :, :]
                feed_dict = {
                    self.tensor.x_input: x_data,
                    self.tensor.y_label: label,
                    self.tensor.keep_prob: self.config.keep_prob
                }
                _, acc, summary, loss = self.sess.run([self.tensor.train_op, self.tensor.acc,
                                                       self.tensor.summary, self.tensor.loss],
                                                      feed_dict=feed_dict)
                loss_list.append(loss)
                acc_list.append(acc)
                file_writer.add_summary(summary, global_step=epoch * steps + step)
                print("step:{}/epoch:{}, loss:{}, acc:{}".format(step, epoch, loss, acc))
            loss_average = sum(loss_list)/len(loss_list)
            loss_summary.append(loss_average)
            acc_average = sum(acc_list)/len(acc_list)
            acc_summary.append(acc_average)
            print("epoch:{}, average loss:{}, average acc:{}".format(epoch, loss_average, acc_average))

            eval_data = self.samples.train_evaluate(9)
            eval_feed = {
                self.tensor.x_input: eval_data,
                self.tensor.keep_prob: 0.
            }
            img = self.sess.run(self.tensor.y_output, feed_dict=eval_feed)
            image_save(img, epoch, self.config.result)

            self.saver.save(self.sess, self.config.model)

        curve_save(acc_summary, loss_summary)

    def predict(self, index):
        eval_data = self.samples.train_evaluate(index)
        eval_feed = {
            self.tensor.x_input: eval_data,
            self.tensor.keep_prob: 0.
        }
        img = self.sess.run(self.tensor.y_output, feed_dict=eval_feed)
        image_save(img, index, self.config.predict_path)


class Samples:
    def __init__(self, config):
        data_path = config.data_path

        self.train_data_path = data_path + '/train/images_512'
        train_data_list = os.listdir(self.train_data_path)
        self.train_data_path_list = [self.train_data_path + '/' + path for path in train_data_list]

        self.train_label_path = data_path + '/train/label_512'
        train_label_list = os.listdir(self.train_label_path)
        self.train_label_path_list = [self.train_label_path + '/' + path for path in train_label_list]

        self.test_data_path = data_path + '/test/images_512'
        test_data_list = os.listdir(self.test_data_path)
        self.test_data_path_list = [self.test_data_path + '/' + path for path in test_data_list]

        self.test_label_path = data_path + '/test/label_512'
        test_label_list = os.listdir(self.test_label_path)
        self.test_label_path_list = [self.test_label_path + '/' + path for path in test_label_list]

    def data(self):
        image = [cv2.imread(_) for _ in self.train_data_path_list]
        label = [cv2.imread(_) for _ in self.train_label_path_list]

        # 提取单通道
        image = [_[:, :, 1] for _ in image]
        label = [_[:, :, 1] for _ in label]

        image = np.asarray(image)
        label = np.asarray(label)

        # 数据预处理
        image = image/255
        label = label/255

        label_ = np.empty([np.asarray(label).shape[0], 512, 512, 2])
        label_[:, :, :, 0] = 1 - label
        label_[:, :, :, 1] = label

        image = np.asarray(image, dtype=np.float32).reshape([-1, 512, 512, 1])
        label_ = np.asarray(label_, dtype=np.float32).reshape([-1, 512, 512, 2])

        return image, label_

    def train_evaluate(self, index):
        path = config.data_path + '/test/images_512/{}.tif'.format(index)
        img = cv2.imread(path)
        img = img[:, :, 1]
        img = np.asarray(img)
        img = img/255
        img = np.asarray(img, dtype=np.float32).reshape([1, 512, 512, 1])
        return img

    @property
    def num(self):
        return 20


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./DRIVE')
    parser.add_argument('--summary_path', type=str, default='./summary')
    parser.add_argument('--result', type=str, default='./result')
    parser.add_argument('--model', type=str, default='./model/model')
    parser.add_argument('--predict_path', type=str, default='./predict')

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--keep_prob', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--decay_step', type=int, default=6000)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    config = parser.parse_args()

    unet = Unet(config)

    if config.train:
        unet.train()
    else:
        for i in range(20):
            unet.predict(i)
