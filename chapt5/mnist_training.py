#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Li Shuai
@Contact   : blockheadls@163.com
@Time    : 2019-03-14 19:33
@File    : mnist_train.py
@Version : v1.0
"""
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

tf.logging.set_verbosity(tf.logging.INFO)

# 参数设置
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
#模型保存的路径
MODEL_SAVE_PATH = '../logs'
MODEL_NAME = 'chapt5.ckpt'

def train(mnist):
    # 定义输入输出的placeholder
    x = tf.placeholder(shape=[None, mnist_inference.INPUT_NODES], dtype=tf.float32, name='x-input')
    y = tf.placeholder(shape=[None, mnist_inference.OUTPUT_NODES], dtype=tf.float32, name='y-output')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y_ = mnist_inference.inference(x, regularizer=regularizer)

    # 定义步数，滑动平均、学习率和优化器都会用到
    global_steps = tf.Variable(0, dtype=tf.int32, trainable=False)

    # 定义滑动平均操作，
    variables_moving_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_steps)
    variables_moving_average_op = variables_moving_average.apply(tf.trainable_variables())

    # 定义损失函数
    cross_enentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.arg_max(y, 1), logits=y_)
    loss = tf.reduce_mean(cross_enentropy) + tf.add_n(tf.get_collection('losses'))

    # 设置学习率
    learing_rate = tf.train.exponential_decay(learning_rate=LEARNING_RATE_BASE,
                                              global_step=global_steps,
                                              decay_rate=LEARNING_RATE_DECAY,
                                              decay_steps=mnist.train.num_examples/BATCH_SIZE)

    # 设置优化方法，这其实是一次完整的前向反馈后向传播
    train_optimize = tf.train.GradientDescentOptimizer(learning_rate=learing_rate).minimize(loss=loss,
                                                                           global_step=global_steps)

    # 评测相关
    # 定义准确度
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 保证每一步训练都要先执行 train_optimize和滑动平均
    with tf.control_dependencies([train_optimize, variables_moving_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, global_step = sess.run([train_op, loss, global_steps], feed_dict={x:xs, y:ys})

            # 每一千轮保存一次模型，评测一次模型
            if i % 1000 == 0:
                acc_ = sess.run(acc, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
                tf.logging.info("After %d training steps, loss on training batch is %g. accuracy on validation is %g."
                                % (global_step, loss_value, acc_))

                saver.save(sess,
                           os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)
    train(mnist)


if __name__=='__main__':
    tf.app.run()

