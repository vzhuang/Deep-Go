import tensorflow as tf
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from parsing.data import DataSetOld

# 12 layer DCNN archiecture (Tian, Zhu 2015)

# kgs = data.read_datasets('data/kgsmedium', 'data/valid', 'data/kgssmalltest', 12)
data = []
positions = []
next_moves = []
for i in range(1000):
    data = np.load('data/kgsmedium/parsed/kgs' + str(i) + '.npy')
    for j in range(len(data)):
        positions.append(data[j][0])
        next_moves.append(data[j][1])
#positions, next_moves = map(list, zip(*data))
class DataSets():
    pass
kgs = DataSets()
kgs.train = DataSetOld(np.rollaxis(np.array(positions), 1, 4), np.array(next_moves))
BOARD_SIZE = 19

sess = tf.InteractiveSession()

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

x_board = tf.reshape(x, [-1, BOARD_SIZE, BOARD_SIZE, 21])
W_conv1 = weight_variable([5, 5, 21, 92])
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

# Final outputs from layer 5
W_convm11 = weight_variable([3, 3, 384, 1])
b_convm11 = bias_variable([1])
h_convm11 = conv2d(h_conv11, W_convm11) + b_convm11

y_conv = tf.sigmoid(tf.reshape(h_convm11, [-1, 361]))

# train
loss = tf.reduce_mean(tf.pow(y_ - y_conv, 2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(1000000):
  batch = kgs.train.next_batch_all(64)
  print(type(batch))
  #print(type(batch[0]), type(batch[1]))
  # print(batch[0].shape)
  # print(batch[0][0].shape)
  # print(batch[0][0][0].shape)
  # print(batch[0][0][0][0])
  # print(batch[1].shape)
  # print(batch[1][0].shape)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: kgs.test.positions, y_: kgs.test.next_moves}))
