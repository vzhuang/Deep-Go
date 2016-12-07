import numpy as np
import sys

class Board():

    def __init__(self, N=19):
        # self.board = np.zeros([N, N])
        self.b_to_move = True
        # self.last_move = None

        # board planes - our/opponent/empty binary
        self.board_planes = []
        self.board_planes.append(np.zeros([19, 19], dtype=np.int8)) # black
        self.board_planes.append(np.zeros([19, 19], dtype=np.int8)) # white
        self.board_planes.append(np.ones([19, 19], dtype=np.int8)) # empty

        # liberty feature planes, black/white resp.
        self.liberty_planes = [np.zeros([3, 19, 19], dtype=np.int8),
                               np.zeros([3, 19, 19], dtype=np.int8)]
        
        self.border_plane = np.zeros([19, 19], dtype=np.int8)
        for i in range(19):
            self.border_plane[0][i] = 1
            self.border_plane[18][i] = 1
            self.border_plane[i][0] = 1
            self.border_plane[i][18] = 1        

        self.curr_chain = set() # we only parse on chain at a time
        self.ko_point = None

    def add_move(self, x, y, c):
        move = (x, y)
        # TODO: handle handicap moves
        # self.board[x, y] = c
        captured_stones = self.check_captures(move, self.get_color())
        for capture in captured_stones:
            self.remove(capture, self.opp_color(self.get_color()))        
        if self.b_to_move:
            self.board_planes[0][move] = 1
            self.board_planes[2][move] = 0
        else:
            self.board_planes[1][move] = 1
            self.board_planes[2][move] = 0
        ind = min(self.num_liberties(move, self.get_color())-1, 2)
        if len(captured_stones) == 1 and ind == 0:
            for stone in captured_stones:
                # should only loop once
                self.ko_point = stone
        else:
            self.ko_point = None

        captured_stones.add(move)
        self.liberty_planes[self.get_color()][ind][move] = 1
        for j in range(3):
            if j != ind:
                self.liberty_planes[self.get_color()][j][move] = 0
        self.update_liberties(captured_stones)
        # self.update_history(move, self.get_color())
        self.last_move = move
        self.change_turn()        

    def get_board_features(self):
        # add all the planes to the input
        inputs = []
        # our/opponent liberties
        if self.b_to_move:
            for plane in self.liberty_planes[0]:
                inputs.append(plane)
            for plane in self.liberty_planes[1]:
                inputs.append(plane)
        else:
            for plane in self.liberty_planes[1]:
                inputs.append(plane)
            for plane in self.liberty_planes[0]:
                inputs.append(plane)
        ko = np.zeros([19, 19])
        if self.ko_point:
            ko[self.ko_point] = 1
        # ko 
        inputs.append(ko)
        inputs.append(self.border_plane)
        return np.copy(inputs)

    def get_legal_moves(self):
        res = np.copy(self.board_planes[2])
        if self.ko_point:
            res[self.ko_point] = 0
        for i in range(19):
            for j in range(19): 
                if res[i, j] and not self.is_legal((i, j), self.get_color()):
                    res[i, j] = 0
        return res    
        
    def remove(self, move, color):
        self.board_planes[color][move] = 0
        self.board_planes[2][move] = 1
        for i in range(3):
            self.liberty_planes[color][i][move] = 0

    def opp_color(self, color):
        return -1 * color + 1

    def get_color(self):
        if self.b_to_move:
            return 0
        return 1

    def change_turn(self):
        if self.b_to_move == True:
            self.b_to_move = False
        else:
            self.b_to_move = True

    def get_neighbors(self, position):
        '''
        Gets all adjacent positions
        '''
        neighbors = []        
        if position[0] > 0:
            neighbors.append((position[0] - 1, position[1]))
        if position[0] < 18:
            neighbors.append((position[0] + 1, position[1]))
        if position[1] > 0:
            neighbors.append((position[0], position[1] - 1))
        if position[1] < 18:
            neighbors.append((position[0], position[1] + 1))
        return neighbors
            
    def update_liberties(self, moves):
        updated = set()
        for move in moves:
            for neighbor in self.get_neighbors(move):
                exists = False
                ncolor = 2
                if self.board_planes[0][neighbor]:
                    exists = True
                    ncolor = 0
                elif self.board_planes[1][neighbor]:
                    exists = True
                    ncolor = 1
                if not exists:
                    continue
                chain = self.get_chain(neighbor, ncolor)
                liberties = self.get_liberties(chain)
                index = min(len(liberties) - 1, 2)
                for stone in chain:
                    if stone not in updated:
                        self.liberty_planes[ncolor][index][stone] = 1
                        for i in range(3):
                            if i != index:
                                self.liberty_planes[ncolor][i][stone] = 0
                        updated.add(stone)

    def num_liberties(self, stone, color):
        '''
        Returns number of liberties in chain of given stone
        '''
        return len(self.get_liberties(self.get_chain(stone, color)))
                    
    def get_liberties(self, chain):
        liberty_set = set()
        for stone in chain:
            for neighbor in self.get_neighbors(stone):
                if self.board_planes[2][neighbor]:
                    liberty_set.add(neighbor)
        return liberty_set
    
    def get_chain(self, stone, color):
        '''
        Floodfill to get all stones in chain of stone
        '''
        self.curr_chain = set()
        self.add_to_chain(stone, color)
        return self.curr_chain

    def add_to_chain(self, stone, color):
        '''
        floodfill curr_chain
        '''
        if stone in self.curr_chain or \
           not self.board_planes[color][stone]:    
            return
        self.curr_chain.add(stone)
        for neighbor in self.get_neighbors(stone):
            self.add_to_chain(neighbor, color)
            
    def check_captures(self, move, color):
        '''
        Returns set of all captured stones by a move
        '''
        captured_stones = set()
        for neighbor in self.get_neighbors(move):
            if self.board_planes[-1*color+1][neighbor] and \
               self.liberty_planes[-1*color+1][0][neighbor]:
                captured_stones = captured_stones.union(
                    self.get_chain(neighbor, -1*color+1))
        return captured_stones

    def is_legal(self, move, color):
        '''
        NOT USED IN TIAN FEATURES
        '''
        # not empty?
        if not self.board_planes[2][move[0]][move[1]]: 
            return False
        # was this point just captured? (ko)
        if move == self.ko_point: # TODO: update ko_point
            return False
        # suicide?
        suicide = True
        for neighbor in self.get_neighbors(move):
            # move is not suicide if:
            # if neighbor empty
            if self.board_planes[2][neighbor]:
                suicide = False
            # if ally neighbor has >1 liberty
            elif self.board_planes[color][neighbor]:
                if self.num_liberties(neighbor, color) > 1:
                    suicide = False
            # if opponent neighbor is in atari
            elif self.board_planes[-1*color+1][neighbor]:
                if self.num_liberties(neighbor, color) == 1:
                    suicide = False
        if suicide:
            return False
        return True
