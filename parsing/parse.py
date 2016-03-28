from multiprocessing.pool import Pool
import pyparsing as pp
import numpy as np
import time
import sys
import os

class Game():
    def __init__(self, rank_b, rank_w, date, num_moves,
                 black_moves, white_moves, handi, handi_moves):
        self._rank_b = rank_b
        self._rank_w = rank_w
        self._date = date
        self._num_moves = num_moves
        self._black_moves = black_moves
        self._white_moves = white_moves
        self._handi = handi
        self._handi_moves = handi_moves

    @property
    def rank_b(self):
        return self._rank_b

    @property
    def rank_w(self):
        return self._rank_w

    @property
    def date(self):
        return self._date

    @property
    def num_moves(self):
        return self._num_moves

    @property
    def black_moves(self):
        return self._black_moves

    @property
    def white_moves(self):
        return self._white_moves

    @property
    def handi(self):
        return self._handi

    @property
    def handi_moves(self):
        return self._handi_moves

def get_games(dir, num_threads = 4):
    '''
    parses all games in a given directory
    '''
    print "Parsing", len(os.listdir(dir)) - 1, 'games' # -1 for .ds_store file
    count = 0
    files = []
    for file in os.listdir(dir):
        if file.endswith(".sgf"):
            files.append(dir + '/' + file)
            count += 1
    p = Pool(processes = num_threads)
    res = p.map_async(parse_games, np.array_split(files, num_threads)).get(1000000)
    games = []
    map(games.extend, res)
    p.close()
    p.join()
    return games

def parse_games(files):
    '''
    Returns parsed version of list of sgf files
    '''
    games = []
    count = 0
    for file in files:
        arr = parse_game(file)
        date, num_moves, handicap, rank_b, rank_w, handi = (0,) * 6
        black_moves = []
        white_moves = []
        handi_moves = []
        index = 0
        while index < len(arr):
            if arr[index] == 'HA':
                index += 2
                handi = int(arr[index])
            elif arr[index] == 'AB': # add handicap stones
                for i in range(handi):
                    if i == 0:
                        index += 2
                    else:
                        index += 3
                    handi_moves.append(parse_move(arr[index]))
            elif arr[index] == 'BR': # black rank
                index += 2
                rank_b = parse_rank(arr[index])
            elif arr[index] == 'WR': # white rank
                index += 2
                rank_w = parse_rank(arr[index])
            elif arr[index]  == 'DT': # date
                index += 2
                date = arr[index]
            elif arr[index] == 'B': # black move
                index += 2
                if arr[index] == ']': # stop if pass
                    break
                black_moves.append(parse_move(arr[index]))
                num_moves += 1
            elif arr[index] == 'W': # white move
                index += 2
                if arr[index] == ']': # stop if pass
                    break
                white_moves.append(parse_move(arr[index]))
                num_moves += 1
            index += 1
        games.append(Game(rank_b, rank_w, date, num_moves,
                          black_moves, white_moves, handi, handi_moves))
    return games    
                            
def parse_move(move):
    '''
    origin is upper left corner, pass = tt
    '''
    indices = "abcdefghijklmnopqrs"
    return (indices.index(move[0]), indices.index(move[1]))
                    
def parse_rank(rank):
    if rank[-1] == 'p':
        return 10 + float(rank[:-1]) / 10 # treat pro rank as 10.x
    elif rank[-1] == 'd':
        return int(rank[:-1])
    elif rank[-1] == 'k':
        return int(rank[:-1]) * -1
    else:
        print 'rank:', rank, 'not valid'
        return None

def parse_game(path):
    '''
    We only care about the main line
    See http://www.britgo.org/tech/sgfspec.html for SGF spec
    Also see http://stackoverflow.com/questions/23163537/handling-escapes-in-pyparsing
    '''
    f = open(path)
    #fstr = f.read().rstrip('\n')
    number = pp.Optional(pp.Literal('+') ^ pp.Literal('-')) + \
             pp.Word(pp.nums)
    escape = pp.Suppress(pp.Literal("\\")) + pp.Word("\\)]:", exact=1)
    text = pp.Combine(pp.ZeroOrMore(escape ^ pp.Regex("[^\\]\\\\:]")))
    real = number + pp.Optional(pp.Literal('.') + pp.Word(pp.nums))
    triple = pp.Literal('1') ^ pp.Literal('2')
    color = pp.Literal('B') ^ pp.Literal('W')
    move = pp.Word('abcdefghijklmnopqrs', exact=2)
    value = pp.Empty() ^ number ^ text ^ real ^ triple ^ color ^ move
    compose = value + pp.Literal(":") + value # handles hyperlinks..
    cvalue = value ^ compose
    prop_ident = pp.Word(pp.alphas.upper(), min = 1) # not exact but works
    prop_value = pp.Literal('[') + cvalue + pp.Literal(']')
    prop = prop_ident + pp.OneOrMore(prop_value)
    node = pp.Literal(';') + pp.ZeroOrMore(prop)
    sequence = pp.ZeroOrMore(node)#node + pp.ZeroOrMore(prop)
    game_tree = pp.Forward()
    # recursively parse sgf
    game_tree << pp.Literal('(') + sequence + pp.ZeroOrMore(game_tree) + pp.Literal(')')
    collection = pp.OneOrMore(game_tree)
    #print fstr
    parsed = collection.parseFile(f)
    f.close()
    return parsed

# start = time.time()
# games = get_games(sys.argv[1], int(sys.argv[2]))
# print time.time()-start
# total = 0
# for game in games:
#     total += game.num_moves
# print total

