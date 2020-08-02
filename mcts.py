#
#
# Python mcts.py
# Created by Willer on 2020/03/27
# I'm in a very, very bad mood today. Feel worthless
#
# Update on 2020/08/02
# Prepare to interview again.
#

import copy
import gc
import random
import numpy as np

class TreeNode:
    """
    MonteCarloTreeSearch TreeNode
    """
    def __init__(self, parent, prior_prob):
        self.parent = parent
        self.prob = prior_prob
        self.q = 0
        self.w = 0
        self.visit = 1
        self.children = {}

    def select(self, c_puct):
        return max(self.children.items(), key=lambda node: node[1].get_value(c_puct))

    def expand(self, action_prob):
        for action, prob in action_prob:
            self.children[action] = TreeNode(self, prob)

    def backup(self, value):
        self.visit += 1
        self.q += value
        # Reverse Value
        if self.parent:
            self.parent.backup(-value)

    def get_value(self, c_puct):
        u = c_puct * self.prob * np.sqrt(self.parent.visit) / (1 + self.visit)
        return self.q / self.visit + u

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None


class MonteCarloTreeSearch:
    """
    MonteCarloTreeSearch implementation
    """
    def __init__(self, policy_value_net, c_puct, n_playout):
        self.root = TreeNode(None, 1.0)
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.policy_value_net = policy_value_net

    def get_action(self, board, tau):
        """
        Sample from visit count
        """
        for i in range(self.n_playout):
            board_copy = copy.deepcopy(board)
            self.playout(board_copy)

        act_visits = [(act, node.visit) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = np.power(visits, 1/tau)
        act_probs /= sum(act_probs)
        return acts, act_probs

    def playout(self, board):
        """
        Execute playout
        """
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            board.move(action)

        action_probs, leaf_value = self.policy_value_net(board)
        end, winner = board.game_end()

        if not end:
            node.expand(action_probs)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0 if winner == board.current_player else -1.0
        node.backup(leaf_value)

    def update_with_move(self, move):
        if move != -1:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)
            gc.collect()


class AlphaPlayer:
    """
    The Player to Collect Data or Play with Human
    """
    def __init__(self, policy_value_net, c_puct, n_playout, is_selfplay=1):
        self.mcts = MonteCarloTreeSearch(policy_value_net, c_puct, n_playout)
        self.is_selfplay = is_selfplay
        self.epoch = 1

    def reset(self):
        self.epoch = 1
        self.mcts.update_with_move(-1)

    def get_action(self, board, tau=1e-3, return_prob=0):

        sensible_moves = board.availables
        return_probs = np.zeros(board.size ** 2)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_action(board, tau)
            return_probs[list(acts)] = probs
            if self.is_selfplay:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            return move, return_probs if return_prob == 1 else move
        else:
            print("WARNING: the board is full")
