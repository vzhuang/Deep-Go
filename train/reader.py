import os
import numpy as np

class BatchIterator():

    def __init__(self, parsed_dir, M=20, batch_size=128, num_channels=8):
        self.parsed_dir = parsed_dir
        self.M = M
        self.batch_size = batch_size
        self.num_channels = 8
        # get number of files in dir
        self.num_files = 0
        for filename in os.listdir(parsed_dir):
            if filename.startswith('board'):
                self.num_files += 1
        self.inds = np.arange(self.num_files)
        np.random.shuffle(self.inds)
        self.batch_inds = np.arange(self.batch_size * self.M)
        np.random.shuffle(self.batch_inds)
        
        self._epochs_completed = 0
        self._set_index_in_epoch = 0
        self.index_in_set = 0
        self.loaded_boards = np.zeros([self.M*self.batch_size, self.num_channels,
                                       19, 19], dtype=np.int8)
        self.loaded_moves = np.zeros([self.M*self.batch_size, 361], dtype=np.int8)  

        # load next batch set
        self.load_next_batch_set()

    def next_batch(self):
        if self.index_in_set >= len(self.loaded_boards):
            self.load_next_batch_set()
        start = self.index_in_set
        end = min(self.index_in_set + self.batch_size, len(self.loaded_boards)-1)
        self.index_in_set += self.batch_size
        return [self.loaded_boards[self.batch_inds[start:end]],
                self.loaded_moves[self.batch_inds[start:end]]]
        

    def load_next_batch_set(self):
        self.index_in_set = 0
        self.loaded_boards = np.zeros([self.M*self.batch_size, self.num_channels,
                                       19, 19], dtype=np.int8)
        self.loaded_moves = np.zeros([self.M*self.batch_size, 361], dtype=np.int8)        
        for i in range(self.M):
            if self._set_index_in_epoch >= self.num_files:
                self._epochs_completed += 1
                self._set_index_in_epoch = 0
                self.inds = np.arange(self.num_files)
                np.random.shuffle(self.inds)
                break
            idx = self.inds[self._set_index_in_epoch]                
            boards_path = self.parsed_dir + 'boards' + str(idx) + '.npy'
            moves_path = self.parsed_dir + 'moves' + str(idx) + '.npy'
            boards = np.load(boards_path)
            moves = np.load(moves_path)
            dim = len(boards)
            start = i*self.batch_size
            self.loaded_boards[start:start+dim] = boards
            self.loaded_moves[start:start+dim] = moves
            self._set_index_in_epoch += 1
        self.batch_inds = np.arange(self.batch_size * self.M)
        np.random.shuffle(self.batch_inds)

    @property    
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch
