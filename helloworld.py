#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by vici on 23/03/2017

import tensorflow as tf


if __name__ == '__main__':
    hello = tf.constant('Hello World!')  # # 常量 生成一个基础 Tensor
    sess1 = tf.Session()
    print(sess1.run(hello))

    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0, tf.float32)
    # print(node1, node2)
    print("session run node1 node2:", sess1.run([node1, node2]))
    node3 = tf.add(node1, node2)
    print("node1: ", node1)
    print("session run node3 to add node1 ,node2: ", sess1.run(node3))
    node4 = tf.constant([[2, 3]])
    print("node4: ", node4)
    print('##########################################')
    w = tf.Variable([.3], tf.float32)   # # 变量
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)    # 占位参数
    liner_mode = w * x + b
    init = tf.global_variables_initializer()  # 初始化tf 内部全局变量
    sess1.run(init)  # 将全局变量添加到当前上下文
    print(sess1.run(liner_mode, {x: [1, 2, 3, 4]}))
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(liner_mode - y)
    loss = tf.reduce_sum(squared_deltas)   # # 上面三步为求 模型精度损失
    print("the old module loss:", sess1.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    fixw = tf.assign(w, [-1])
    fixb = tf.assign(b, [1])  # 修改变量
    sess1.run([fixb, fixw])
    print("The fixed module loss: ", sess1.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    # ------------------梯度下降训练模型-----------------
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)
    fixw2 = tf.assign(w, [1])
    fixb2 = tf.assign(b, [2])  # 修改变量
    sess1.run([fixw2, fixb2])
    print("module loss before training: ", sess1.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
    for i in range(1000):
        sess1.run(train, {x: [1, 2, 3, 4], y: [0, 1, 2, 3]})

        print(sess1.run([w, b]))









