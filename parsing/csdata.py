import parse
import numpy as np
import time
import sys
import os
from multiprocessing.pool import Pool

class DataSet():

    def __init__(self, files):
        self.files = np.array(files)
        self._epochs_completed = 0
        self._index_in_epoch = 0        

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    def next_batch_random(self, dir, num_games, batch_size):
        def load_game(dir, ind):
            return np.load(dir + 'kgs' + str(ind) + '.npy')
        inds = np.random.choice(np.arange(num_games), size=16, replace=False)
        games = np.array([load_game(ind) for ind in inds])
        games.flatten()
        return np.random.choice(games, batch_size, replace=False)
        
    def next_batch_sim(self, dir, num_games, batch_size, num_threads=8):
        # inds = np.random.choice(np.arange(num_games), size=16, replace=False)
        # moves = []
        # for i in range(8):
        #     game_moves = split_game(self.files[inds[i]])
        #     if game_moves is not None:
        #         moves.extend(split_game(self.files[inds[i]]))
        # return zip(*np.array(moves)[np.random.choice(np.arange(len(moves)),
        #                                              size=batch_size,
        #                                              replace=False)])
        
        inds = np.random.choice(np.arange(num_games), size=32, replace=False)
        to_parse = self.files[inds]
        split_files = [to_parse[i:i+num_threads] for i in range(0, len(to_parse), num_threads)]
        p = Pool(processes=num_threads)
        pairs = p.map_async(run_batch_thread, zip(range(num_threads), split_files)).get(10000000)
        games = [j for i in pairs for j in i]
        for i, game in enumerate(games):
            games[i] = game[int(0.9*np.random.random()):]
            
        moves = [move for game in games for move in game]
        p.close()
        p.join()
        res = np.array(moves)[np.random.choice(np.arange(len(moves)),
                                               size=batch_size,
                                               replace=False)]
        for i, tup in enumerate(res):
            flip = np.random.randint(2)
            rot = np.random.randint(4)
            cube = np.zeros([8, 19, 19])
            for j in range(8):
                if flip:
                    tup[0][j] = np.fliplr(tup[0][j])
                tup[0][j] = np.rot90(tup[0][j], rot)
                cube[j] = tup[0][j]
            move = tup[1].reshape([19, 19])
            if flip:
                move = np.fliplr(move)
            move = np.rot90(move, rot)
            res[i] = (cube, move.flatten())
        return zip(*res)        
        
    def next_batch_all(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self.index_in_epoch > self._num_points:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle
            perm = np.arange(self._num_points)
            np.random.shuffle(perm)
            self._positions = self._positions[perm]
            self._next_moves = self._next_moves[perm]
            # start the next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_points
        end = self._index_in_epoch
        return self._positions[start:end], self._next_moves[start:end]


class DataSetOld():

    def __init__(self, positions, moves):
        self._positions = positions
        self._next_moves = moves
        self._num_points = len(positions)
        self._epochs_completed = 0
        self._index_in_epoch = 0        

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def index_in_epoch(self):
        return self._index_in_epoch
        
    def next_batch_all(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self.index_in_epoch > self._num_points:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle
            perm = np.arange(self._num_points)
            np.random.shuffle(perm)
            self._positions = self._positions[perm]
            self._next_moves = self._next_moves[perm]
            # start the next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_points
        end = self._index_in_epoch
        return self._positions[start:end], self._next_moves[start:end]        

class GameParser():
    '''
    Parses Game object into Clark/Storkey features
    '''
    def __init__(self, game):
        self.game = game
        self.last_move = None
        self.b_to_move = True
        self.b_move = 0
        self.w_move = 0

        # board planes - our/opponent/empty binary
        self.board_planes = []
        self.board_planes.append(np.zeros([19, 19], dtype=np.int8)) # black
        self.board_planes.append(np.zeros([19, 19], dtype=np.int8)) # white
        self.board_planes.append(np.ones([19, 19], dtype=np.int8)) # empty

        # liberty feature planes (binary true/false)
        self.liberty_planes = [np.zeros([3, 19, 19], dtype=np.int8),
                               np.zeros([3, 19, 19], dtype=np.int8)]
        
        self.border_plane = np.zeros([19, 19], dtype=np.int8)
        for i in range(19):
            self.border_plane[0][i] = 1
            self.border_plane[18][i] = 1
            self.border_plane[i][0] = 1
            self.border_plane[i][18] = 1

        self.curr_chain = set() # we only parse one chain at a time
        self.ko_point = None

    def extract_data_points(self):
        '''
        Iterates through game and returns all position, next move pairs
        '''
        data_points = []
        if self.game.handi:
            for move in self.game.handi_moves:
                self.update_liberties([move])
                self.add_move(move)
            self.b_to_move = False

        # ignore SGFs that are too short
        if self.game.num_moves < 20:
            return None
        # capture size should be unchanged by handicap stones
            
        for i in range(self.game.num_moves):
        
            # add all the planes to the input
            inputs = []
            # our/opponent liberties
            if self.b_to_move:
                for plane in self.liberty_planes[0]:
                    inputs.append(plane)
                    # print plane.T
                for plane in self.liberty_planes[1]:
                    inputs.append(plane)
                    # print plane.T
            else:
                for plane in self.liberty_planes[1]:
                    inputs.append(plane)
                    # print plane.T
                for plane in self.liberty_planes[0]:
                    inputs.append(plane)
                    # print plane.T
            ko = np.zeros([19,19])
            if self.ko_point:
                ko[self.ko_point] = 1
            # ko 
            inputs.append(ko)
            inputs.append(self.border_plane)
            
            move = (20, 20)
            if self.b_to_move:
                move = self.game.black_moves[self.b_move]
                self.b_move += 1
            else:
                move = self.game.white_moves[self.w_move]
                self.w_move += 1

            # TODO: get rid of onehot - we can always convert later
            # onehot = np.zeros(361)
            # onehot[move[0]*19 + move[1]] = 1
            onehot = np.zeros([19, 19])
            onehot[move] = 1
            onehot = onehot.flatten()
            data_points.append((np.copy(inputs), onehot))

            # check for captures, update next few moves
            captured_stones = self.check_captures(move, self.get_color())
            for capture in captured_stones:
                self.remove(capture, self.opp_color(self.get_color()))
            # play next move
            self.add_move(move)
            # TODO: check next line logic
            # essentially, next move can affect liberties of
            # adjacent stones, need to check            
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

        return data_points

    def get_closest_color(row, col):
        closest_b = 1

    def exp(self, plane, param = 0.1):
        return np.exp(-param * plane)

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

    def add_move(self, move):
        '''
        adds moves to board planes, updates legality
        '''
        if self.b_to_move:
            self.board_planes[0][move] = 1
            self.board_planes[2][move] = 0
        else:
            self.board_planes[1][move] = 1
            self.board_planes[2][move] = 0
            
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
                if num_liberties(neighbor, color) > 1:
                    suicide = False
            # if opponent neighbor is in atari
            elif self.board_planes[-1*color+1][neighbor]:
                if num_liberties(neighbor, color) == 1:
                    suicide = False
        if suicide:
            return False
        return True

def run_batch_thread(args):
    res = []
    for file_name in args[1]:
        game_moves = split_game(file_name)
        if game_moves is not None:
            res.append(game_moves)
    return res        

def get_datapoints(dir, num_threads = 4):    
    count = 0
    files = []
    for file in os.listdir(dir):
        if file.endswith(".sgf"):
            files.append(dir + '/' + file)
            count += 1
    print count, 'games'
    p = Pool(processes = num_threads)
    data = []
    count = 0
    failed = 0
    for parsed in p.imap_unordered(split_game, files):
        if parsed is None:
            failed += 1
        else:
            np.save(dir + '/parsed' + '/kgs' + str(count), parsed)
            count += 1
    print(failed, count)
    p.close()
    p.join()
    # positions, next_moves = map(list, zip(*data))
#    return DataSet(np.rollaxis(np.array(positions), 1, 4), np.array(next_moves))

def split_game(file):
    game = parse.parse_games(file)
    gp = GameParser(game)
    return gp.extract_data_points()
 
def split_games(games):
    dps = []
    for game in games:
        gp = GameParser(game)
        dps.extend(gp.extract_data_points())
    return dps   
                
def read_datasets(train_dir, valid_dir, test_dir, num_threads = 4):
    class DataSets():
        pass
    data_sets = DataSets()    
    data_sets.train = get_datapoints(train_dir, num_threads)
    data_sets.valid = get_datapoints(valid_dir, num_threads)
    data_sets.test = get_datapoints(test_dir, num_threads)    
    return data_sets
    
