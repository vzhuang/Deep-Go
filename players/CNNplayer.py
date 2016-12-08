#!/usr/bin/env python
import sys
import os
import re
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..') 
import numpy as np
import tensorflow as tf
from models.cnn import TZ_CNN
from player import Player
from gtp import GTP

ckpt_path = '/home/vincent/Documents/Projects/Deep-Go/saved/model.ckpt'

def softmax(x, temp=1.0):
    return np.exp(x / temp) / np.sum(np.exp(x / temp), axis=0)

class CNNPlayer(Player):

    def __init__(self, model):
        self.model = model
        # build tf evaluation graph
        saver = tf.train.Saver(tf.trainable_variables())

        self.sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())

        saver.restore(self.sess, ckpt_path)

    def gen_move(self, num_channel=8, pick_best=True):
        # parse into feature planes
        x_in = np.rollaxis(np.array([self.board.get_board_features()]), 1, 4)
        # compute model logits

        logits = self.sess.run(self.model.logits, feed_dict={self.model.x: x_in})[0]

        # old_stdout = sys.stdout
        # log_file = open('/home/vincent/Documents/Projects/Deep-Go/movelog.txt',"w")
        
        # sys.stdout = log_file
        # print(logits)
        # sys.stdout = old_stdout
        
        # zero out illegal moves
        
        legal = self.board.get_legal_moves()
        for i in range(19):
            for j in range(19):
                if not legal[i, j]:
                    logits[19*i+j] = 0                    
        if not pick_best:
            probs = softmax(logits)
            best = np.random.choice(range(361), p=probs)
            return (best / 19, best % 19)
        best = np.argmax(logits)
        return (best / 19, best % 19)

    def name(self):
        return "CNN Player"

    def version(self):
        return "1.0"

if __name__ == '__main__':
    model = TZ_CNN(8, 19)
    cnn = CNNPlayer(model)
    gtp = GTP(cnn)
    gtp.handler()
