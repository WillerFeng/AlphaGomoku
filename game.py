#
#
# Python game.py
# Created by Willer on 2020/03/25
# Still no interview message, I may have to change to C++ or Swift :(
#
#

import numpy as np
from itertools import count

class Board():

    def __init__(self, size=7):
        self.size = size
        self.states = {}
        self.players = [1, 2]

    def init_board(self, start_player=0):

        self.current_player = self.players[start_player]  # start player
        self.availables = list(range(self.size ** 2))
        self.states = {}
        self.last_move = -1

    def current_state(self, channel=7):
        """
        return the board state from the perspective of the current player.
        state shape: 7 * self.size * self.size
        """
        square_state = np.zeros((channel, self.size, self.size))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            try:
                for i in range(channel-1):
                    if i & 1 == 0:
                        square_state[i][divmod(move_curr[i//2],self.size)] = 1.0
                    else:
                        square_state[i][divmod(move_oppo[i//2],self.size)] = 1.0
            except IndexError:
                pass
        if len(self.states) % 2 == 0:
            square_state[-1][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):

        if self.last_move == -1:
            return False, -1
        h, w = divmod(self.last_move, self.size)
        if any(len(set(self.states.get(i, -1) for i in range(self.last_move-j*self.size, self.last_move+(5-j)*self.size))) == 1 for j in range(5)):
            return True, self.current_player
        if any(len(set(self.states.get(self.last_move//self.size*self.size+i, -1) for i in range(j, j+5))) == 1 for j in range(max(0, self.last_move%self.size-4), min(self.size-4, self.last_move%self.size+1))):
            return True, self.current_player
        if any(len(set(self.states.get(i, -1) for i in range(self.last_move-j*(self.size+1), self.last_move+(5-j)*(self.size+1), self.size+1))) == 1 for j in range(5)):
            return True, self.current_player
        if any(len(set(self.states.get(i, -1) for i in range(self.last_move-j*(self.size-1), self.last_move+(5-j)*(self.size-1), self.size-1))) == 1 for j in range(5)):
            return True, self.current_player
        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    def __init__(self, board):
        self.board = board

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
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, tau=1e-3):

        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        for t in count():
            move, move_probs = player.get_action(self.board, tau=tau, return_prob=1)

            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.do_move(move)

            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()

            if end:
                winners_z = np.zeros(len(current_players), dtype=float)
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0

                player.reset()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return zip(states, mcts_probs, winners_z), t+1
