# -- encoding:utf-8 --
"""
Created by MengCheng Ren on 2019/5/29
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def acc_compute(target, logistic):
    target = tf.reshape(target, [-1, 2])
    logistic = tf.reshape(logistic, [-1, 2])
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(target, axis=-1), tf.argmax(logistic, axis=-1)), tf.float32))
    return acc


def curve_save(acc, loss):
    plt.figure()
    plt.plot(np.arange(len(loss)), loss)
    plt.title('LOSS')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(['loss'], loc="upper right")
    plt.savefig('./' + "LOSS.png")

    plt.figure()
    plt.plot(np.arange(len(acc)), acc)
    plt.title('acc')
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend(loc="lower right")
    plt.savefig('./' + "ACC.png")
    # plt.show()


def image_save(img, index, save_path):
    arr1 = img[0, :, :, 0]
    arr1[np.where(arr1 >= 0.5)] = 1
    arr1[np.where(arr1 < 0.5)] = 0

    arr1 = np.reshape(arr1, [512, 512]) * 255
    arr1 = arr1.astype(np.uint8)

    cv2.imwrite(save_path + '/{}.png'.format(index), arr1)