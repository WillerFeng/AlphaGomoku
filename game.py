#
#
# Python game.py
# Created by Willer on 2020/03/25
# Still no interview message, I may have to change to C++ or Swift :(
#
#
#
# Game: 1. self.play(player) return [[state, action, value]]
# Construct the environment
#
#

import numpy as np
import copy
from itertools import count
from collections import defaultdict, deque

class Board:

    def __init__(self, size=7, recent_lens=7):
        self.size = size
        self.gomoku = 4
        self.players = [1, 2]
        self.recent_lens = recent_lens
        self.current_state = None

    def init_board(self, start_player=0):

        self.last_move = -1
        self.current_player = self.players[start_player]
        self.availables = list(range(self.size ** 2))
        self.states = np.zeros(shape=(self.size, self.size), dtype=int)
        self.current_state = copy.deepcopy(self.states)

    def move(self, move):
        x, y = divmod(move, self.size)
        self.states[x][y] = 1
        self.states *= -1
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.current_state = copy.deepcopy(self.states)
        self.last_move = move

    def has_a_winner(self):

        vl = vr = 0
        hu = hd = 0
        dh = dl = 0
        rdh = rdl = 0

        if self.last_move == -1:
            return False
        x, y = divmod(self.last_move, self.size)
        for i in range(y+1, self.size):
            if self.states[x][i] == self.states[x][y]:
                vr += 1
            else:
                break
        for i in range(y-1, -1, -1):
            if self.states[x][i] == self.states[x][y]:
                vl += 1
            else:
                break
        if vr + vr >= self.gomoku-1:
            return True

        for i in range(x+1, self.size):
            if self.states[i][y] == self.states[x][y]:
                hd += 1
            else:
                break
        for i in range(x-1, -1, -1):
            if self.states[i][y] == self.states[x][y]:
                hu += 1
            else:
                break
        if hd + hu >= self.gomoku-1:
            return True


        for i in range(1, 4):
            if x+i == self.size or y+i == self.size:
                break
            if self.states[x+i][y+i] == self.states[x][y]:
                dl += 1
            else:
                break
        for i in range(1, 4):
            if x-i == -1 or y-i == -1:
                break
            if self.states[x-i][y-i] == self.states[x][y]:
                dh += 1
            else:
                break
        if dl + dh >= self.gomoku-1:
            return True

        for i in range(1, 4):
            if x+i == self.size or y-i == -1:
                break
            if self.states[x+i][y-i] == self.states[x][y]:
                rdl += 1
            else:
                break
        for i in range(1, 4):
            if x-i == -1 or y+i == self.size:
                break
            if self.states[x-i][y+i] == self.states[x][y]:
                rdh += 1
            else:
                break
        if rdl + rdh >= self.gomoku-1:
            return True

        return False

    def game_end(self):
        win = self.has_a_winner()
        if win:
            return True, self.players[0] if self.current_player == self.players[1] else self.players[1]
        elif not len(self.availables):
            return True, -1
        return False, -1



class Game(object):
    def __init__(self, board):
        self.board = board
        self.games_count = 0

    def graphic(self, board, player1, player2):
        self.size = board.size
        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(self.size):
            print("{0:4}".format(x), end='')
        print('\r\n')
        for i in range(self.size - 1, -1, -1):
            print("{0:2d}".format(i), end='')
            for j in range(self.size):
                loc = i * self.size + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(4), end='')
                elif p == player2:
                    print('O'.center(4), end='')
                else:
                    print('_'.center(4), end='')
            print('\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        start a game between two players
        """
        pass

    def start_self_play(self, player, is_shown=0, tau=1e-3):

        self.games_count += 1
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs = [], []

        move_trajectory = []
        if self.games_count % 50 == 0:
            for t in count():
                move, move_probs = player.get_action(self.board, tau=tau, return_prob=1)

                states.append([self.board.current_state])
                mcts_probs.append(move_probs)

                self.board.move(move)
                move_trajectory += [move]

                if is_shown:
                    self.graphic(self.board, p1, p2)
                end, winner = self.board.game_end()

                if end:
                    winners_z = np.ones(len(states), dtype=float)
                    if winner != -1:
                        winners_z[-2::-2] = -1.0

                    with open('record.txt', 'a+') as f:
                        f.write(str(move_trajectory)+'\n\r')

                    player.reset()
                    if is_shown:
                        if winner != -1:
                            print("Game end. Winner is player:", winner)
                        else:
                            print("Game end. Tie")
                    return zip(states, mcts_probs, winners_z), t+1

        else:
            for t in count():
                move, move_probs = player.get_action(self.board, tau=tau, return_prob=1)

                states.append(list(copy.deepcopy(self.board.current_state)))
                mcts_probs.append(move_probs)
                self.board.move(move)

                if is_shown:
                    self.graphic(self.board, p1, p2)
                end, winner = self.board.game_end()

                if end:
                    winners_z = np.ones(len(states), dtype=float)
                    if winner != -1:
                        winners_z[-2::-2] = -1.0

                    player.reset()
                    if is_shown:
                        if winner != -1:
                            print("Game end. Winner is player:", winner)
                        else:
                            print("Game end. Tie")
                    return zip(states, mcts_probs, winners_z), t+1
