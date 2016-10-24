import tensorflow as tf
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from parsing.data import DataSet

# 12 layer DCNN archiecture (Tian, Zhu 2015)

# kgs = data.read_datasets('data/kgsmedium', 'data/valid', 'data/kgssmalltest', # 12)
# data = []
# positions = []
# next_moves = []
# for i in range(1000):
#     data = np.load('data/kgsmedium/parsed/kgs' + str(i) + '.npy')
#     for j in range(len(data)):
#         positions.append(data[j][0])
#         next_moves.append(data[j][1])
#positions, next_moves = map(list, zip(*data))

files = []
for file_name in os.listdir('data/kgstrain'):
    if file_name.endswith('.sgf'):
        files.append('data/kgstrain/' + file_name)
class DataSets():
    pass
kgs = DataSets()
kgs.train = DataSet(files)

BOARD_SIZE = 19

sess = tf.InteractiveSession()

# placeholders
x = tf.placeholder(tf.float32, [None, 19, 19, 23])
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

x_board = tf.reshape(x, [-1, BOARD_SIZE, BOARD_SIZE, 23])
W_conv1 = weight_variable([5, 5, 23, 92])
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

# Final outputs from layer 5
W_convm5 = weight_variable([3, 3, 384, 1])
b_convm5 = bias_variable([1])
h_convm5 = conv2d(h_conv5, W_convm5) + b_convm5

y_conv = tf.sigmoid(tf.reshape(h_convm5, [-1, 361]))

# train
loss = tf.reduce_mean(tf.pow(y_ - y_conv, 2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
curr = time.time()
for i in range(1, 1000000):
  batch = kgs.train.next_batch_sim('data/kgstrain/', 144000, 128)
  batch[0] = np.array(batch[0])
  batch[1] = np.array(batch[1])
  batch[0] = np.rollaxis(batch[0], 1, 4)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print("time for 100 minibatches:", time.time() - curr)
    curr = time.time()
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: kgs.test.positions, y_: kgs.test.next_moves}))
