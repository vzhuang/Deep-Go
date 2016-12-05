import gc
import os
import parse
import numpy as np
from csdata import GameParser
from contextlib import closing
from multiprocessing.pool import Pool


def split_game(file):
    game = parse.parse_games(file)
    gp = GameParser(game)
    return gp.extract_data_points()


def process_datapoints(data_dir, out_dir, games_per_thread=500, num_threads=4, moves_per_file=128):    
    files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".sgf"):
            files.append(data_dir + '/' + filename)
    print('preprocessing', len(files), 'games')
    # shuffle sgfs
    np.random.shuffle(files)
    blk = games_per_thread * num_threads    
    for i in range((len(files) / blk) + 1):
        print('processed', i * blk, 'SGFs')
        process_block(files, i, blk, out_dir, num_threads, moves_per_file)
         # del data
         # del boards
         # del moves
         # gc.collect()
         # p.close()
         # p.terminate()
         # p.join()

def process_block(files, i, blk, out_dir, num_threads, moves_per_file):
    with closing(Pool(processes=num_threads)) as p:
        # p = 
        blk_files = files[i*blk:min((i+1)*blk, len(files)-1)]
        data = []
        for parsed in p.imap_unordered(split_game, blk_files):
            data.append(parsed)
        boards = []
        moves = []
        for game in data:
            if game is None:
                continue
            for move in game:
                boards.append(move[0])
                moves.append(move[1])
        inds = np.arange(len(boards))
        np.random.shuffle(inds)
        boards = np.array(boards, dtype=np.int8)
        moves = np.array(moves, dtype=np.int8)
        for j in range((len(boards) / moves_per_file)):
            # for b in boards[j*moves_per_file:min((j+1)*moves_per_file, len(boards)-1)]:
            #     print(b.shape)
            in_ind = inds[j*moves_per_file:min((j+1)*moves_per_file, len(boards)-1)]
            w_boards = np.stack(boards[in_ind])
            w_moves = np.stack(moves[in_ind])

            np.save(out_dir+'/boards'+str(i*blk + j), w_boards)
            np.save(out_dir+'/moves'+str(i*blk + j), w_moves)

if __name__ == '__main__':
    # process_datapoints('/home/vincent/Documents/Projects/Deep-Go/data/test',
    #                    '/home/vincent/Documents/Projects/Deep-Go/parsed/test',
    #                    games_per_thread=20,
    #                    num_threads=10)        
    process_datapoints('/home/vincent/Documents/Projects/Deep-Go/data/kgstrain',
                       '/home/vincent/Documents/Projects/Deep-Go/parsed/kgstrain',
                       games_per_thread=200,
                       num_threads=10)    
    # process_datapoints('~/Documents/Projects/Deep-Go/data/kgstrain',
    #                    '~/Documents/Projects/Deep-Go/parsed/kgstrain',
    #                    num_threads=10)
