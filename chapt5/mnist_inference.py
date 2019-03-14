#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Li Shuai
@Contact   : blockheadls@163.com
@Time    : 2019-03-14 18:24
@File    : mnist_inference.py
@Version : v1.0
"""

import tensorflow as tf

INPUT_NODES = 784
OUTPUT_NODES = 10
LAYER1_NODES = 500


# 使用get_variable训练时创建变量，预测时使用变量，不同layer的变量在不同的域下面。
# 预测时可以方便的获得变量的滑动平均值。
# 训练时使用变量自身，预测时使用变量的滑动平均值。
def get_weight_varibale(shape, regularizer):
    # truncated_normal_initializer 舍弃两个标准差的数据，重新生成数据
    weights = tf.get_variable('weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 将规则化的变量加到'losses'集合中，为之后的loss计算做准备
    # regularizer(weights) 表示权重矩阵的范数
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 构建网络结构
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights1 = get_weight_varibale([INPUT_NODES, LAYER1_NODES], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODES], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1)+biases)

    with tf.variable_scope("layer2"):
        weights2 = get_weight_varibale([LAYER1_NODES, OUTPUT_NODES], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODES], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights2) + biases

    return layer2

