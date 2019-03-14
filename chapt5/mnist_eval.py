#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Li Shuai
@Contact   : blockheadls@163.com
@Time    : 2019-03-14 23:27
@File    : mnist_eval.py
@Version : v1.0
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_training

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(shape=[None, mnist_inference.INPUT_NODES], name='x-input', dtype=tf.float32)
        y = tf.placeholder(shape=[None, mnist_inference.OUTPUT_NODES], name='y-output', dtype=tf.float32)

        y_ = mnist_inference.inference(x, regularizer=None)

        correction_predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        acc = tf.reduce_mean(tf.cast(correction_predict, tf.float32))

        # 使用滑动平均后的变量来预测输出
        # variables_to_restore函数可以将变量的滑动平均值赋给变量
        # 测试集acc是 0.9839
        variables_moving_average = tf.train.ExponentialMovingAverage(mnist_training.MOVING_AVERAGE_DECAY)
        variables_to_restore = variables_moving_average.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 不使用滑动平均值, 测试集是0.9836
        # saver = tf.train.Saver()

        with tf.Session() as sess:
            # 获取最新的模型
            ckpt = tf.train.get_checkpoint_state(mnist_training.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                # saver.restore(sess, '../logs/chapt5.ckpt-27001')
                acc_ = sess.run(acc, feed_dict={x:mnist.test.images, y: mnist.test.labels})
                tf.logging.info("The path of loading model is %s ." % ckpt.model_checkpoint_path)
                tf.logging.info("Acc is %g ." % acc_)


def main(argv=None):
    mnist_data = input_data.read_data_sets("../data/mnist/", one_hot=True)
    evaluate(mnist_data)


if __name__ == '__main__':
    tf.app.run()

