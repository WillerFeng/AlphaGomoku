#
#
# Python mcst.py
# Created by Willer on 2020/03/26
#
#

import pickle
from game import Board, Game
from mcts import AlphaPlayer
from network import PolicyValueNet
import argparse

class HumanPlayer():

    def __init__(self, board_size):
        self.board_size = board_size
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            x, y = map(int, input("Your Move : ").split())
            move = x * self.board.size + y
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move


def run(args):

    model_file = 'train_model'
    try:
        board = Board(board_size=args.size)
        game = Game(board)

        policy = PolicyValueNet(args.size)
        policy.load(model_file)

        alpha_player = AlphaPlayer(policy.policy_value_fn, c_puct=5, n_playout=args.playout)
        human_player = HumanPlayer(args.size)

        game.start_play(human, alpha_player, start_player=args.start_player, is_shown=1)

    except ModuleNotFoundError:
        print("Policy Model File not Found")

    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model type")
    parser.add_argument('-s','--size', default='7')
    parser.add_argument('-e','--epoch', default='100')
    parser.add_argument('-p','--playout', default='800')
    parser.add_argument('-start','--start_player', default='1')
    args = parser.parse_args()
    run(args)
