import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..') 

import time
import numpy as np
import tensorflow as tf
from models.cnn import TZ_CNN, SmallCNN
from reader import BatchIterator

ckpt_path = '/home/vincent/Documents/Projects/Deep-Go/saved/model.ckpt'

def train_step(loss, lr):
    return tf.train.AdamOptimizer(lr).minimize(loss)

def loss_funcs(logits, y_):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                                                         y_))
    correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return loss, accuracy

    
def train(model, data_dir, valid_dir, num_epochs, num_channels, batch_size,
          games_per_set, lr, restore=False, checkpoint_path=ckpt_path):
    logits = model.logits
    y_ = model.y_
    loss, accuracy = loss_funcs(logits, y_)
    loss_summ = tf.scalar_summary("loss", loss)
    train_op = train_step(loss, lr)
    
    kgs = BatchIterator(data_dir, games_per_set, batch_size, num_channels)

    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver(tf.trainable_variables())
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.train.SummaryWriter('logs', graph=sess.graph)    
    
    if restore:
        saver.restore(sess, ckpt_path)
        
    step = 0
    curr = time.time()
    
    while kgs.epochs_completed < num_epochs:
        step += 1
        batch = kgs.next_batch()
        indices = np.zeros([len(batch[0])])
        for j in range(len(batch[0])):
            indices[j] = np.argmax(batch[1][j])
        batch[0] = np.rollaxis(batch[0], 1, 4)

        _, curr_loss, summary_str = sess.run([train_op, loss, summary_op],
                                             feed_dict={model.x: batch[0],
                                                        model.y_: indices})
        summary_writer.add_summary(summary_str, step)
        if step % 100 == 0:
            train_accuracy = accuracy.eval(session=sess,
                                           feed_dict={model.x: batch[0],
                                                      model.y_: indices})
            print("step %d, training accuracy %g"%(step, train_accuracy))
            print("time for 100 minibatches:", time.time() - curr)
            curr = time.time()

        if step % 10000 == 0:
            saver.save(sess, checkpoint_path)
            # compute validation error
            print('computing validation error...')
            valid_start = time.time()
            valid = BatchIterator(valid_dir,
                                  games_per_set,
                                  batch_size,
                                  num_channels)
            valid_accuracy = []
            while valid.epochs_completed < 1:
                batch = valid.next_batch()
                indices = np.zeros([len(batch[0])])
                for j in range(len(batch[0])):
                    indices[j] = np.argmax(batch[1][j])
                batch[0] = np.rollaxis(batch[0], 1, 4)
                valid_accuracy.append(accuracy.eval(session=sess,
                                                    feed_dict={model.x:batch[0],
                                                               model.y_: indices}))
            print('validation accuracy:', np.mean(valid_accuracy))
            print('time for validation:', time.time() - valid_start)

if __name__ == '__main__':
    model = TZ_CNN(8, 19)
    data_dir = '/home/vincent/Documents/Projects/Deep-Go/parsed/kgstrain/'
    valid_dir = '/home/vincent/Documents/Projects/Deep-Go/parsed/kgsvalid/'
    train(model, data_dir, valid_dir, 20, 8, 128, 25, 0.00025)
    
