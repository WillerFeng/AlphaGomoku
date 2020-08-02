#
#
# Python network.py
# Created by Willer on 2020/03/28
# Swift is hard as C++. I give up again. :(
#
#

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class ResidualBlock(nn.Module):
    def __init__(self, inner_channel, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
            nn.Conv2d(inner_channel, inner_channel, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
            nn.Conv2d(inner_channel, inner_channel, kernel_size, stride, padding, dilation, bias=False),
        )

    def forward(self, x):
        out = self.block(x)
        out += x
        return out

class PolicyValueNet(nn.Module):
    """
    policy-value network
    """
    def __init__(self, board_size, feature_channel=7, channel=48):
        super(PolicyValueNet, self).__init__()

        self.size = board_size
        self.conv1 = nn.Conv2d(feature_channel, channel, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(channel, kernel_size=3, padding=1)
        self.block2 = ResidualBlock(channel, kernel_size=3, padding=1)
        self.block3 = ResidualBlock(channel, kernel_size=3, padding=1)

        self.act_conv1 = nn.Conv2d(channel, 2, kernel_size=1)
        self.act_fc = nn.Linear(2 * self.size ** 2, self.size ** 2)

        self.val_conv1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.val_fc1 = nn.Linear(self.size ** 2, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):

        x = self.conv1(state_input)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 2 * self.size ** 2)
        x_act = F.softmax(self.act_fc(x_act), dim=1)

        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, self.size ** 2)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))
        return x_act, x_val


class AlphaAgent:
    """
    The Reinforcement Learning Agent
    """
    def __init__(self, board_size, feature_channel=7, model_file=None):

        self.l2_const = 1e-4
        self.lr = 2e-4
        self.board_size = board_size
        self.feature_channel = feature_channel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_value_net = PolicyValueNet(self.board_size, self.feature_channel).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=self.lr, weight_decay=self.l2_const)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma = 0.95)

    def policy_value_fn(self, board):

        legal_positions = board.availables
        current_state = torch.FloatTensor([board.current_state]).to(self.device)
        act_probs, value = self.policy_value_net(current_state)

        act_probs = act_probs.data.cpu().numpy().flatten()
        act_probs = zip(legal_positions, act_probs[legal_positions])

        value = value.data[0][0]
        return act_probs, value

    def train(self, state_batch, mcts_probs, winner_batch):

        state_batch  = torch.FloatTensor(state_batch).to(self.device)
        mcts_probs   = torch.FloatTensor(mcts_probs).to(self.device)
        winner_batch = torch.FloatTensor(winner_batch).to(self.device)

        act_probs, value = self.policy_value_net(state_batch)

        # loss = (z - v)^2 - \pi^T * log(p) + c||theta||^2
        value_loss  = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(torch.log(mcts_probs + 1e-5) * act_probs, 1))
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return value_loss.item(), policy_loss.item()


    def save(self, filename):
        torch.save(self.policy_value_net.state_dict(), filename + "_policy_value_net_" + str(datetime.datetime.now()))
        torch.save(self.optimizer.state_dict(), filename + "_optimizer_" + str(datetime.datetime.now()))

    def load(self, filename):
        self.policy_value_net.load_state_dict(torch.load(filename + "_policy_value_net"))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
