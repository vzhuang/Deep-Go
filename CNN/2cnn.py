import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from parsing import data

# Two layer convnet with x filters of some specified size

kgs = data.read_datasets('data/kgstrain', 'data/valid', 'data/kgstest', 12)
BOARD_SIZE = 19

sess = tf.InteractiveSession()

# num_filters = int(sys.argv[1])
# filter_size = int(sys.argv[2])

# placeholders
x = tf.placeholder(tf.float32, [None, 19, 19, 21])
y_ = tf.placeholder(tf.float32, [None, 361])

# helper functions for weights/biases in CNN
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# zero padded
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# convolution layer
# x_board = tf.reshape(x, [-1, 19, 19, 21])
# W = weight_variable([5, 5, 21, 64])
# b = bias_variable([64])
# h_conv = tf.nn.relu(conv2d(x_board, W) + b)

# W2 = weight_variable([3, 3, 64, 64])
# b2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_conv, W2) + b2)

# W_m2 = weight_variable([3, 3, 64, 1])
# b_m2 = bias_variable([1])
# h_convm2 = conv2d(h_conv2, W_m2) + b_m2

# y_conv = tf.sigmoid(tf.reshape(h_convm2, [-1, 361]))

x_board = tf.reshape(x, [-1, BOARD_SIZE, BOARD_SIZE, 21])
W_conv1 = weight_variable([5, 5, 21, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

W_conv4 = weight_variable([5, 5, 64, 48])
b_conv4 = bias_variable([48])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

W_conv5 = weight_variable([5, 5, 48, 48])
b_conv5 = bias_variable([48])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

# Final outputs from layer 5
W_convm5 = weight_variable([5, 5, 48, 1])
b_convm5 = bias_variable([1])
h_convm5 = conv2d(h_conv5, W_convm5) + b_convm5

y_conv = tf.sigmoid(tf.reshape(h_convm5, [-1, 361]))

# # fully connected layer
# W_fc = weight_variable([64 * num_filters, 100])
# B_fc = bias_variable([100])
# h_conv_flat = tf.reshape(h_conv, [-1, 64 * num_filters])
# h_fc = tf.nn.relu(tf.matmul(h_conv_flat, W_fc) + b_fc)

# # softmax output layer with dropout
# keep_prob = tf.placeholder("float")
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# W_fc2 = weight_variable([100, 64])
# b_fc2 = bias_variable([64])
# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train
loss = -tf.reduce_mean(tf.pow(y_ - y_conv, 2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(1000000):
  batch = kgs.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: kgs.test.positions, y_: kgs.test.next_moves}))
