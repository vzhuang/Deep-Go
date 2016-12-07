import sys
import re

def coords_from_str(s):
    x = ord(s[0].upper()) - ord('A')
    if x >= 9: x -= 1
    y = int(s[1:])
    y -= 1
    return x,y

def str_from_coords(x, y):
    if x >= 8: x += 1
    return chr(ord('A')+x) + str(y+1)

class GTP():
    """
    Lightweight GTP implementation roughly based on
    https://github.com/pasky/michi/blob/master/michi.py

    Used for interfacing with GoGUI, KGS, etc.
    """
    def __init__(self, player):
        self.player = player

    def handler(self):
        commands = ['boardsize', 'clear_board', 'genmove', 'komi',
                    'list_commands', 'name', 'play', 'version']
        color_dict = {'b': 0, 'w': 1}
        while True:
            line = sys.stdin.readline().strip()
            if line == 0:
                return
            command = [s.lower() for s in line.split()]
            if re.match('\d+', command[0]):
                cmdid = command[0]
                command = command[1:]
            else:
                cmdid = ''
            # owner_map = W*W*[0]
            ret = ''            
            # print('Got command %s', line.split())
            if command[0] == 'boardsize':
                # assume board size is 19 for simplicity here
                if int(command[1]) != 19:
                    print('Incorrect board size')
                    ret = None
            elif command[0] == 'name':
                ret = self.player.name()
            elif command[0] == 'version':
                ret = self.player.version()
            elif command[0] == 'clear_board':
                self.player.clear_board()
            elif command[0] == 'play':
                # assume x,y is valid move
                x, y = coords_from_str(command[2])
                self.player.add_move(x, y, color_dict[command[1]])
                # TODO: handle pass
                
            elif command[0] == 'genmove':
                # TODO: handle pass
                # TODO: handle resign
                # CURRENT: assume play move
                coords = self.player.gen_move()
                self.player.add_move(coords[0], coords[1], 0)
                ret = str_from_coords(*coords)
            elif command[0] == 'komi':
                self.player.set_komi(float(command[1]))
            elif command[0] == 'list_commands':
                ret = '\n'.join(commands)
            else:
                # print('unknown command')
                ret = None
            if ret is not None:
                print('=%s %s\n\n' % (cmdid, ret,))
            else:
                print('?%s ???\n\n' % (cmdid,))
            sys.stdout.flush()
