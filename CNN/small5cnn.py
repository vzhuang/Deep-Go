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
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# zero padded
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

x_board = tf.reshape(x, [-1, BOARD_SIZE, BOARD_SIZE, 8])
W_conv1 = weight_variable([5, 5, num_channels, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_board, W_conv1) + b_conv1)

# _activation_summary(h_conv1)

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

# _activation_summary(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

# _activation_summary(h_conv3)

W_conv4 = weight_variable([5, 5, 64, 48])
b_conv4 = bias_variable([48])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

# _activation_summary(h_conv4)

W_conv5 = weight_variable([5, 5, 48, 48])
b_conv5 = bias_variable([48])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

# _activation_summary(h_conv5)

# Final outputs from layer 5
W_convm5 = weight_variable([5, 5, 48, 1])
b_convm5 = bias_variable([1])
h_convm5 = conv2d(h_conv5, W_convm5) + b_convm5

# y_conv = tf.sigmoid(tf.reshape(h_convm5, [-1, 361]))
y_conv = tf.reshape(h_convm5, [-1, 361])

# train
# loss = tf.reduce_mean(tf.pow(y_conv - y_, 2))
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
  # train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  # evaluate test accuracy on this dataset
  # big_batch = kgs.test.all()    

# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: kgs.test.positions, y_: kgs.test.next_moves}))
