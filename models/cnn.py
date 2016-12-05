import tensorflow as tf
import numpy as np
import sys
import os
import re
import time

# sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..') 

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

# zero padded
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

class TZ_CNN():
    
    def __init__(self, num_channels, BOARD_SIZE=19):
        self.x = tf.placeholder(tf.float32, [None, 19, 19, num_channels])
        x_board = tf.reshape(self.x, [-1, BOARD_SIZE, BOARD_SIZE, num_channels])
        self.y_ = tf.placeholder(tf.int64, [None])

        W_conv1 = weight_variable([5, 5, num_channels, 92])
        b_conv1 = bias_variable([92])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        W_conv2 = weight_variable([5, 5, 92, 384])
        b_conv2 = bias_variable([384])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        W_conv3 = weight_variable([5, 5, 384, 384])
        b_conv3 = bias_variable([384])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

        W_conv4 = weight_variable([5, 5, 384, 384])
        b_conv4 = bias_variable([384])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

        W_conv5 = weight_variable([5, 5, 384, 384])
        b_conv5 = bias_variable([384])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

        W_conv6 = weight_variable([5, 5, 384, 384])
        b_conv6 = bias_variable([384])
        h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)

        W_conv7 = weight_variable([5, 5, 384, 384])
        b_conv7 = bias_variable([384])
        h_conv7 = tf.nn.relu(conv2d(h_conv6, W_conv7) + b_conv7)

        W_conv8 = weight_variable([5, 5, 384, 384])
        b_conv8 = bias_variable([384])
        h_conv8 = tf.nn.relu(conv2d(h_conv7, W_conv8) + b_conv8)

        W_conv9 = weight_variable([5, 5, 384, 384])
        b_conv9 = bias_variable([384])
        h_conv9 = tf.nn.relu(conv2d(h_conv8, W_conv9) + b_conv9)

        W_conv10 = weight_variable([5, 5, 384, 384])
        b_conv10 = bias_variable([384])
        h_conv10 = tf.nn.relu(conv2d(h_conv9, W_conv10) + b_conv10)

        W_conv11 = weight_variable([5, 5, 384, 384])
        b_conv11 = bias_variable([384])
        h_conv11 = tf.nn.relu(conv2d(h_conv10, W_conv11) + b_conv11)

        W_convm11 = weight_variable([3, 3, 384, 1])
        b_convm11 = bias_variable([1])
        h_convm11 = conv2d(h_conv11, W_convm11) + b_convm11

        self.logits = tf.reshape(h_convm11, [-1, BOARD_SIZE**2])

class SmallCNN():
    def __init__(self, num_channels, BOARD_SIZE=19):
        self.x = tf.placeholder(tf.float32, [None, 19, 19, num_channels])
        x_board = tf.reshape(self.x, [-1, BOARD_SIZE, BOARD_SIZE, num_channels])
        self.y_ = tf.placeholder(tf.int64, [None])
        
        W_conv1 = weight_variable([5, 5, num_channels, 92])
        b_conv1 = bias_variable([92])
        h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)

        W_conv2 = weight_variable([5, 5, 92, 384])
        b_conv2 = bias_variable([384])
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

        W_conv3 = weight_variable([5, 5, 384, 384])
        b_conv3 = bias_variable([384])
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

        W_conv4 = weight_variable([5, 5, 384, 384])
        b_conv4 = bias_variable([384])
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

        W_conv5 = weight_variable([5, 5, 384, 384])
        b_conv5 = bias_variable([384])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

        W_convm5 = weight_variable([3, 3, 384, 1])
        b_convm5 = bias_variable([1])
        h_convm5 = conv2d(h_conv5, W_convm5) + b_convm5

        self.logits = tf.reshape(h_convm5, [-1, BOARD_SIZE**2])
