#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by vici on 23/03/2017

import tensorflow as tf
import numpy as np


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




