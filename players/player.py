#!/usr/bin/env python 
 
import numpy as np
from board import Board
from gtp import GTP

class Player(object):
    def __init__(self):
        self.board = Board()

    def set_komi(self, komi):
        self.komi = komi

    def clear_board(self):
        self.board = Board()

    def name(self):
        return "Player name"

    def version(self):
        return "1.0"

    def add_move(self, x, y, color):
        self.board.add_move(x, y, color)
        
class RandomPlayer(Player):

    def __init__(self):
        super(Player, self).__init__()

    def gen_move(self):
        legal = self.board.get_legal_moves()
        possible = []
        for i in range(19):
            for j in range(19):
                if legal[i, j]:
                    possible.append((i,j))
        to_play = possible[np.random.randint(0, len(possible))]
        return to_play

if __name__ == "__main__":
    rp = RandomPlayer()
    gtp = GTP(rp)
    gtp.handler()
        
    
