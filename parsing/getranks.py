import sys
import os

import pyparsing as pp

def get_games(dir):
    '''
    parses all games in a given directory
    '''
    print "Parsing", len(os.listdir(dir)) - 1, 'games' # -1 for .ds_store file
    games = []
    count = 0
    ranks = {}
    for file in os.listdir(dir):
        if file.endswith(".sgf"):
            count += 1
            print count
            arr = parse_game(dir + '/' + file)
            date, num_moves, handicap, rank_b, rank_w, handi = (0,) * 6
            index = 0
            while index < len(arr):
                if arr[index] == 'BR': # black rank
                    index += 2
                    rank_b = parse_rank(arr[index])
                    if rank_b in ranks.keys():
                        ranks[rank_b] += 1
                    else:
                        ranks[rank_b] = 1
                elif arr[index] == 'WR': # white rank
                    index += 2
                    rank_w = parse_rank(arr[index])
                    if rank_w in ranks.keys():
                        ranks[rank_w] += 1
                    else:
                        ranks[rank_w] = 1
                index += 1
    for key in ranks.keys():
        print key, ranks[key]                            
                    
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

get_games(sys.argv[1])


