import random
import numpy as np
import datetime
import gc
import torch
from collections import defaultdict, deque
from game import Board, Game
from mcts import AlphaPlayer
from network import PolicyValueNet, AlphaAgent
from tensorboardX import SummaryWriter

class AlphaSelfTrain():
    def __init__(self):

        self.size = 7
        self.board = Board(size=self.size)
        self.game = Game(self.board)

        self.learn_rate = 1e-3
        self.tau = 1.0
        self.n_playout = 10
        self.c_puct = 5
        self.buffer_size = 100000
        self.batch_size = 16

        self.train_epochs = 1
        self.check_freq = 5
        self.game_total = 200
        self.game_single = 2

        self.train_epoch = 1
        self.train_games = 1

        self.summary = SummaryWriter()
        self.buffer = deque(maxlen=self.buffer_size)

        self.agent = AlphaAgent(self.size)
        self.alpha_player = AlphaPlayer(self.agent.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)


    def collect_selfplay_data(self):

        for i in range(self.game_single):

            data, game_lens = self.game.start_self_play(self.alpha_player, tau=self.tau)
            data = list(data)
            self.buffer.extend(data)

            self.summary.add_scalar('game_lens/train', game_lens, self.train_games)
            self.train_games += 1

    def learn(self):

        for i in range(self.train_epochs):

            mini_batch = random.sample(self.buffer, self.batch_size)
            state_batch, mcts_probs_batch, winner_batch = zip(*mini_batch)
            value_loss, policy_loss = self.agent.train(state_batch, mcts_probs_batch, winner_batch)

            self.summary.add_scalar('loss/value', value_loss, self.train_epoch)
            self.summary.add_scalar('loss/policy', policy_loss, self.train_epoch)
            self.train_epoch += 1

    def train(self):
        try:
            for i in range(self.game_total):
                print('epoch:', i)
                print('     self-play')
                self.collect_selfplay_data()

                if len(self.buffer) > self.batch_size:
                    self.learn()
                    print('     update')
                # if (i+1) % self.check_freq == 0:
                #     torch.save(self.agent.policy_value_net.state_dict(), '')

                gc.collect()
                torch.cuda.empty_cache()
        except KeyboardInterrupt:
            print('\n\rquit')
        self.summary.close()

if __name__ == '__main__':
    agent = AlphaSelfTrain()
    agent.train()
