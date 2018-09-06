#coding:utf-8
'''
Created on 2018年8月26日

@author: Administrator
'''

import tensorflow as tf
import tensorflow.contrib.layers as layers

def get_weight(shape,regularizer):
    W=tf.Variable(initial_value=tf.random_normal(shape), dtype=tf.float32)
    
    
    #正则化权重W
    tf.add_to_collection("losses", layers.l2_regularizer(regularizer)(W))
    
    return W

def get_bias(shape):
    
    b=tf.Variable(initial_value=tf.constant(0.01,shape=shape))
    return b

def forward(X, regularizer):
    
    W_layer1=get_weight([2,11], regularizer)
    B_layer1=get_bias([11])
    Y_layer1=tf.nn.relu(tf.matmul(a=X, b=W_layer1)+B_layer1)
    
    W_layer2=get_weight([11,1], regularizer)
    B_layer2=get_bias([1])
    Y_predicted=tf.matmul(a=Y_layer1, b=W_layer2)+B_layer2
    
    return Y_predicted
