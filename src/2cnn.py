import tensorflow as tf
import numpy as np
import wthor
import sys

# Two layer convnet with x filters of some specified size

sess = tf.InteractiveSession()

num_filters = int(sys.argv[1])
filter_size = int(sys.argv[2])

othello = wthor.read_data()

# placeholders
x = tf.placeholder(tf.float32, [None, 64])
y = tf.placeholder(tf.float32, [None, 64])

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

# typically no pooling for games (see Silver 2015)
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# convolution layer
x_image = tf.reshape(x, [-1, 8, 8, 1])
W = weight_variable([filter_size, filter_size, 1, num_filters])
b = bias_variable([num_filters])
h_conv = tf.nn.relu(conv2d(x_image, W) + b)
# h = max_pool_2x2(h)

# fully connected layer
W_fc = weight_variable([64 * num_filters, 100])
B_fc = bias_variable([100])
h_conv_flat = tf.reshape(h_conv, [-1, 64 * num_filters])
h_fc = tf.nn.relu(tf.matmul(h_conv_flat, W_fc) + b_fc)

# softmax output layer with dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([100, 64])
b_fc2 = bias_variable([64])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# train
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(1000):
  batch = othello.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
from parsing import data


