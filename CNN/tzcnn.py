import tensorflow as tf
import numpy as np
import sys
import os
import re
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..') 

TOWER_NAME = 'tower'

from train.reader import BatchIterator

num_channels = 8

kgs = BatchIterator('/home/vincent/Documents/Projects/Deep-Go/parsed/kgstrain/', 25, 128, num_channels)

BOARD_SIZE = 19

sess = tf.InteractiveSession()

# placeholders
x = tf.placeholder(tf.float32, [None, 19, 19, num_channels])
# y_ = tf.placeholder(tf.float32, [None, 361])
y_ = tf.placeholder(tf.int64, [None])

# helper functions for weights/biases in CNN
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

# zero padded
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

x_board = tf.reshape(x, [-1, BOARD_SIZE, BOARD_SIZE, num_channels])
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

# y_conv = tf.sigmoid(tf.reshape(h_convm11, [-1, 361]))
y_conv = tf.reshape(h_convm11, [-1, 361])

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)
correct_prediction = tf.equal(tf.argmax(y_conv,1), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
curr = time.time()
loss_summ = tf.scalar_summary("loss", loss)

# Add histograms for trainable variables.
for var in tf.trainable_variables():
  tf.histogram_summary(var.op.name, var)

# add op for merging summary
summary_op = tf.merge_all_summaries()

# add Saver ops
saver = tf.train.Saver()

summary_writer = tf.train.SummaryWriter('logs', graph=sess.graph)

for i in range(1, 1000000):
  # batch = kgs.train.next_batch_sim('data/kgstrain/', 144000, 128)
  batch = kgs.next_batch()
  # print(batch[0].shape)
  # print(batch[1].shape)
  # batch[0] = np.array(batch[0])
  # batch[1] = np.array(batch[1])
  # we don't need one hot for this??
  indices = np.zeros([len(batch[0])])  
  # replace border mask with all ones mask  
  for j in range(len(batch[0])):
      batch[0][j][7] = np.ones([19, 19])
      indices[j] = np.argmax(batch[1][j])
  batch[0] = np.rollaxis(batch[0], 1, 4)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: indices})
    outputs = y_conv.eval(feed_dict={
        x:batch[0], y_: indices})
    for arr in outputs:
        print(np.max(arr))
    print("step %d, training accuracy %g"%(i, train_accuracy))
    print("time for 100 minibatches:", time.time() - curr)
    curr = time.time()
  _, curr_loss, summary_str = sess.run([train_step, loss, summary_op],
                                       feed_dict={x: batch[0], y_: indices})
  summary_writer.add_summary(summary_str, i)
