import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..') 

import time
import numpy as np
import tensorflow as tf
from models.cnn import TZ_CNN, SmallCNN
from train.reader import BatchIterator

ckpt_path = '/home/vincent/Documents/Projects/Deep-Go/saved/model.ckpt'

def loss_funcs(logits, y_):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_))
    correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return loss, accuracy
    
def eval(model, test_dir, num_channels, batch_size, games_per_set):
    logits = model.logits
    y_ = model.y_
    loss, accuracy = loss_funcs(logits, y_)
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    test_start = time.time()
    test = BatchIterator(test_dir, games_per_set, batch_size, num_channels)
    test_accuracy = []
    while test.epochs_completed < 1:
        batch = test.next_batch()
        indices = np.zeros([len(batch[0])])
        for j in range(len(batch[0])):
            indices[j] = np.argmax(batch[1][j])
        batch[0] = np.rollaxis(batch[0], 1, 4)
        test_accuracy.append(accuracy.eval(session=sess,
                                            feed_dict={model.x:batch[0],
                                                       model.y_: indices}))
    print('test accuracy:', np.mean(test_accuracy))
    print('time for computing:', time.time() - test_start)

if __name__ == '__main__':
    model = SmallCNN(8, 19)
    test_dir = '/home/vincent/Documents/Projects/Deep-Go/parsed/kgstest/'
    eval(model, test_dir, 8, 128, 25)
    
