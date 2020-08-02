#
#
# Python train.py
# Created by Willer on 2020/03/27
# Update on 2020/08/02
#

import random
import numpy as np
import datetime
import gc
import torch
import argparse
from collections import defaultdict, deque
from game import Board, Game
from mcts import AlphaPlayer
from network import PolicyValueNet, AlphaAgent, TEAgent
from tensorboardX import SummaryWriter


class AlphaSelfTrain:
    def __init__(self, type):

        self.size = 7
        self.board = Board(size=self.size)
        self.game = Game(self.board)

        self.tau = 1.0
        self.n_playout = 64
        self.c_puct = 5
        self.feature_channel = 1

        self.learn_rate = 2e-4
        self.buffer_size = 400
        self.batch_size = 128

        self.check_freq = 5
        self.game_total_collect = 400
        self.game_singl_collect = 10
        self.train_singl_epoch  = 2

        self.train_epoch = 1
        self.games_epoch = 1
        self.buffer  = deque(maxlen=self.buffer_size)

        self.type = type
        if self.type == 'RAW':
            self.alpha_agent  = AlphaAgent(self.size, self.feature_channel)
            self.summary = SummaryWriter('runs/AlphaAgent_' + str(datetime.datetime.now()))
        elif self.type == 'TE':
            self.alpha_agent  = TEAgent(self.size, self.feature_channel)
            self.summary = SummaryWriter('runs/AlphaAgent-with-TE_' + str(datetime.datetime.now()))
        else:
            raise NameError('Unknown Type')
        self.alpha_player = AlphaPlayer(self.alpha_agent.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)


    def collect_selfplay_data(self):

        for i in range(self.game_singl_collect):

            data, game_lens = self.game.start_self_play(self.alpha_player, tau=self.tau)
            data = list(data)
            self.buffer.extend(data)

            if self.type == 'TE':
                self.alpha_agent.buffer.add(data[-1][0], data[-1][1], data[-1][2])

            self.summary.add_scalar('game_lens/train', game_lens, self.games_epoch)
            self.games_epoch += 1


    def learn(self):

        for i in range(self.train_singl_epoch):

            mini_batch = random.sample(self.buffer, self.batch_size)
            state_batch, mcts_probs_batch, winner_batch = zip(*mini_batch)
            value_loss, policy_loss = self.alpha_agent.train(state_batch, mcts_probs_batch, winner_batch)

            self.summary.add_scalar('loss/value', value_loss, self.train_epoch)
            self.summary.add_scalar('loss/policy', policy_loss, self.train_epoch)
            self.train_epoch += 1


    def train(self):
        try:
            for i in range(self.game_total_collect):
                print('epoch:', i)
                print('     self-play')
                self.collect_selfplay_data()

                if len(self.buffer) > self.batch_size:
                    self.learn()
                    print('     update')

                gc.collect()
                torch.cuda.empty_cache()

                # if (i+1) % self.check_freq == 0:
                #     torch.save(self.alpha_agent.policy_value_net.state_dict(), '')

        except KeyboardInterrupt:
            print('\n\rquit')
        self.summary.close()

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_random_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default='RAW', help='agent type')
    args = parser.parse_args()
    agent = AlphaSelfTrain(args.type)
    agent.train()
