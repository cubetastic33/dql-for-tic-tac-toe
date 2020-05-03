from collections import namedtuple
from itertools import count
import time
import random
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


def render_board(cells):
    for row in cells:
        print("|".join([[" ", "X", "O"][cell] for cell in row]))


class TicTacToe:
    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        """Resets the game"""
        self.cells = torch.zeros((3, 3), device=self.device, dtype=torch.long)
        self.x_turn = random.choice([True, False])

    def get_state(self, player_is_x=True):
        cells = self.cells.flatten()
        player = 2 * player_is_x - 1
        return (
            torch.cat((cells == player, cells == -player, cells == 0))
            .float()
            .unsqueeze(0)
        )

    def perform_action(self, action):
        """Performs an action"""
        row = action // 3
        column = action % 3
        if self.cells[row][column] != 0:
            # The chosen cell is already occupied
            return -10, False
        self.cells[row][column] = 2 * self.x_turn - 1
        if (result := self.eval()) is not None:
            # The game is over
            # You can never lose if the game ends immediately after an action
            # The rewards are flipped because we give it to the other player
            return [-1, -5][result], True
        # The action has been completed and the game is still in progress
        self.x_turn = not self.x_turn
        return 0, False

    def render(self):
        """Renders the board"""
        render_board(self.cells)

    def eval(self):
        """Checks if the game is over and if it has, who has won"""
        # A player can only win if the last move was made by them
        # So if the next move is to be X's, then the player is -1, otherwise it's 1
        player = 2 * self.x_turn - 1
        winning_line = [player] * 3
        # Diagonals
        lines = [self.cells.diagonal().tolist(), self.cells.flip(1).diagonal().tolist()]
        # Rows
        lines.extend(self.cells.tolist())
        # Columns
        lines.extend(self.cells.T.tolist())
        if winning_line in lines:
            return player
        if 0 not in self.cells:
            # There are no empty cells left, and nobody's won, so it's a draw
            return 0
        # The game is still in progress
        return None


def select_action(state, policy_net):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    return torch.tensor([[random.randrange(9)]], device=DEVICE, dtype=torch.long)


class HAL9000(nn.Module):
    def __init__(self):
        # Call parent class's __init__()
        super().__init__()

        self.fc1 = nn.Linear(in_features=27, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=9)

    def forward(self, x):
        """Performs a forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x


Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, *args):
        """Saves an experience"""
        if len(self.memory) < self.capacity:
            # We still haven't filled up the memory even once
            self.memory.append(None)
        # Save the experience
        self.memory[self.index] = Experience(*args)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        """Return batch_size random experiences from memory"""
        if len(self.memory) < batch_size:
            # If we still don't have as many as batch_size experiences stored
            return None
        return random.sample(self.memory, batch_size)


def optimize_model(memory, policy_net, target_net, optimizer, losses):
    if experiences := memory.sample(BATCH_SIZE):
        batch = Experience(*zip(*experiences))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_states_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        illegal_mask = torch.tensor(
            tuple(map(lambda e: e.state[0][18 + e.action[0]].item() != 1, experiences)),
            device=DEVICE,
            dtype=torch.bool,
        )
        loss_mask = torch.tensor(
            tuple(map(lambda r: r.item() == -5, batch.reward)),
            device=DEVICE,
            dtype=torch.bool,
        )
        win_mask = torch.tensor(
            tuple(map(lambda r: r.item() == 10, batch.reward)),
            device=DEVICE,
            dtype=torch.bool,
        )
        draw_mask = torch.tensor(
            tuple(map(lambda r: r.item() == -1, batch.reward)),
            device=DEVICE,
            dtype=torch.bool,
        )
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for the next states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        next_state_values = target_net(next_states_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = next_state_values * GAMMA + reward_batch
        expected_state_action_values[illegal_mask] = -10.0
        expected_state_action_values[loss_mask] = -5.0
        expected_state_action_values[win_mask] = 10.0
        expected_state_action_values[draw_mask] = -1.0

        # Compute loss
        loss = F.mse_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        losses.append(loss.item())


DEVICE = torch.device("cpu")
BATCH_SIZE = 128
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 200
TARGET_UPDATE = 10
SAVE_FREQUENCY = 500
SAVE_FILE = "HAL9000.pt"

env = TicTacToe(DEVICE)
episode = 0
steps_done = 0
policy_net = HAL9000().to(DEVICE)
optimizer = optim.SGD(policy_net.parameters(), lr=0.01)
memory = ReplayMemory(5000)

if os.path.isfile(SAVE_FILE):
    checkpoint = torch.load(SAVE_FILE)
    episode = checkpoint["episode"]
    steps_done = checkpoint["steps_done"]
    memory = checkpoint["memory"]
    policy_net.load_state_dict(checkpoint["policy_net"])
    optimizer.load_state_dict(checkpoint["optimizer"])

target_net = HAL9000().to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

results = []
loss = []

start_time = time.thread_time()

for i in count(episode):
    # Initialize the environment and state
    env.reset()
    state_x = env.get_state()
    state_o = env.get_state(False)
    action_x = None
    action_o = None
    done = False
    for t in count():
        if env.x_turn:
            # Play as X
            action_x = select_action(state_x, policy_net)
            reward, done = env.perform_action(action_x.item())
            if reward != -10:
                if done:
                    results.append(0 if reward == -1 else 1)
                    reward_x = torch.tensor([10 if reward == -5 else -1], device=DEVICE)
                    memory.push(state_x, action_x, env.get_state(), reward_x)
                    optimize_model(memory, policy_net, target_net, optimizer, loss)
                next_state_o = env.get_state(False)
                if action_o is not None:
                    # If O has also made a move after this game started
                    reward_o = torch.tensor([reward], device=DEVICE)
                    # Save the experience in memory
                    memory.push(state_o, action_o, next_state_o, reward_o)
                    optimize_model(memory, policy_net, target_net, optimizer, loss)
                state_o = next_state_o
            else:
                # Illegal move
                reward_x = torch.tensor([reward], device=DEVICE)
                memory.push(state_x, action_x, state_x, reward_x)
                optimize_model(memory, policy_net, target_net, optimizer, loss)
        else:
            # Play as O
            action_o = select_action(state_o, policy_net)
            reward, done = env.perform_action(action_o.item())
            if reward != -10:
                if done:
                    results.append(0 if reward == -1 else -1)
                    reward_o = torch.tensor([10 if reward == -5 else -1], device=DEVICE)
                    memory.push(state_o, action_o, env.get_state(False), reward_o)
                    optimize_model(memory, policy_net, target_net, optimizer, loss)
                next_state_x = env.get_state()
                if action_x is not None:
                    # If X has also made a move after this game started
                    reward_x = torch.tensor([reward], device=DEVICE)
                    # Save the experience in memory
                    memory.push(state_x, action_x, next_state_x, reward_x)
                    optimize_model(memory, policy_net, target_net, optimizer, loss)
                state_x = next_state_x
            else:
                # Illegal move
                reward_o = torch.tensor([reward], device=DEVICE)
                memory.push(state_o, action_o, state_o, reward_o)
                optimize_model(memory, policy_net, target_net, optimizer, loss)

        if done:
            break
    if len(results) == SAVE_FREQUENCY:
        print(i + 1, "episodes completed")
        print(f"{time.thread_time() - start_time} seconds")
        print("  X  |  -  |  O  ")
        print(
            f" {str(results.count(1)).zfill(3)} | {str(results.count(0)).zfill(3)} | {str(results.count(-1)).zfill(3)}"
        )
        print(
            "Mean loss in last",
            SAVE_FREQUENCY,
            "episodes:",
            round(torch.tensor(loss, device=DEVICE).mean().item(), 4),
        )
        print()
        checkpoint = {
            "episode": i + 1,
            "steps_done": steps_done,
            "memory": memory,
            "policy_net": policy_net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, SAVE_FILE)
        # Uncomment for loss graphs
        # plt.ion()
        # plt.figure(2)
        # plt.clf()
        # plt.title("Loss")
        # plt.plot(loss)
        # plt.ioff()
        # plt.show()
        results = []
        loss = []
        start_time = time.thread_time()
    # Update the target networks with the policy networks every TARGET_UPDATE episodes
    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
