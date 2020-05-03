import sys
import random
import itertools
import math
import random
import time
import pickle
from itertools import count
from collections import defaultdict, namedtuple

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QStackedWidget,
    QVBoxLayout,
    QBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
)
from PySide2 import QtCore


# Just there so that we can import the policy net from the checkpoint
Experience = namedtuple("Experience", ("state", "action", "next_state", "reward"))


# Just there so that we can import the policy net from the checkpoint
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity


# The network
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


class GameBoard(QWidget):
    def __init__(self):
        # Call parent class's __init__()
        super().__init__()

        # Create board
        board = QGridLayout()
        board.setSpacing(12)
        # Loop for creating the cells
        for j in range(9):
            # Create cell
            cell = QPushButton(" ")
            cell.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            cell.setObjectName(f"{j % 3}|{j // 3}")
            # Add click callback
            cell.clicked.connect(self.conquer_cell)
            # Set cursor to be a pointer when on the cell
            cell.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            # Add the cell to the board
            board.addWidget(cell, j // 3, j % 3)

        # Set the screen's layout to be the QGridLayout we created
        self.main_layout = board
        self.setLayout(self.main_layout)
        self.setObjectName("gameBoard")
        self.x_turn = True
        self.device = torch.device("cpu")
        self.policy_net = HAL9000().to(self.device)
        self.policy_net.load_state_dict(torch.load("HAL9000.pt")["policy_net"])
        self.policy_net.eval()

    # Function to conquer a cell
    def conquer_cell(self):
        # If the cell is not already occupied, and the game isn't over yet
        if self.sender().text() == " " and self.check_game_status() is None:
            # Fill the cell with the player's character
            self.sender().setText("X")
            if (game_status := self.check_game_status()) == 0:
                self.parentWidget().game_status.setText("It's a tie!")
            elif game_status == 1:
                self.parentWidget().game_status.setText("X has won!")
            else:
                self.play_o()
                if (game_status := self.check_game_status()) == 0:
                    self.parentWidget().game_status.setText("It's a tie!")
                elif game_status == 1:
                    self.parentWidget().game_status.setText("O has won!")

    def play_o(self):
        with torch.no_grad():
            output = self.policy_net(self.get_state().float()).to(self.device)[0]
        for action in output.argsort(descending=True):
            cell = self.findChildren(QPushButton)[action.item()]
            if cell.text() != " ":
                print("chose a filled cell")
            else:
                cell.setText("O")
                break
        return

    def get_state(self):
        cells = []
        for cell in self.findChildren(QPushButton):
            cells.append({" ": 0, "X": 1, "O": -1}[cell.text()])
        player = -1
        cells = torch.tensor(cells, device=self.device)
        return (
            torch.cat((cells == player, cells == -player, cells == 0))
            .float()
            .unsqueeze(0)
        )

    # Function to check if someone's won
    def check_game_status(self):
        # Create a dictionary to store the X and O positions
        occupied_cells = {"X": defaultdict(set), "O": defaultdict(set)}
        # Loop over all pairs of cells
        for pair in itertools.combinations(np.array(self.findChildren(QPushButton)), 2):
            # Save the player occupying the first cell and make sure it's not empty
            # Also check if both cells in the pair are occupied by the same player
            if (player := pair[0].text()) == pair[1].text() != " ":
                # Store the coordinates of both the cells
                c0 = np.array([int(i) for i in pair[0].objectName().split("|")])
                c1 = np.array([int(i) for i in pair[1].objectName().split("|")])
                # Step is the difference between the position vectors of the two cells
                step = c1 - c0
                # Add the cells to the dictionary
                occupied_cells[player][step.tostring()] |= {tuple(c0), tuple(c1)}
                # If at least three points have the same step value amongst themselves
                if len(occupied_cells[player][step.tostring()]) >= 3:
                    # Create a dictionary to store the root cells
                    roots = defaultdict(list)
                    # Loop over all the cells with that step
                    for point in occupied_cells[player][step.tostring()]:
                        root_cell = compute_root_cell(np.array(point), step)
                        roots[root_cell.tostring()].append(point)
                        # If 3 of these cells have the same root, this player has won
                        if len(roots[root_cell.tostring()]) == 3:
                            for winning_point in roots[root_cell.tostring()]:
                                coordinates = "|".join([str(i) for i in winning_point])
                                self.findChild(QPushButton, coordinates).setStyleSheet(
                                    "background-color: #a2fafa"
                                )
                            return 1 if player == "X" else -1
        if " " not in [cell.text() for cell in self.findChildren(QPushButton)]:
            # Nobody has won and the board is full, so return a tie
            return 0
        return None


def compute_root_cell(cell, step):
    while ((cell - step) >= 0).all():
        cell -= step
    return cell


class GameScreen(QWidget):
    def __init__(self):
        # Call parent class's __init__()
        super().__init__()

        self.board = GameBoard()
        self.board.setFixedSize(600, 600)

        self.game_status = QLabel()
        self.game_status.setObjectName("gameStatus")
        self.game_status.setMaximumHeight(30)
        self.game_status.setAlignment(QtCore.Qt.AlignCenter)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_game)
        self.reset_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.reset_button.setMinimumWidth(120)
        self.exit_button = QPushButton("Exit Game")
        self.exit_button.clicked.connect(sys.exit)
        self.exit_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.exit_button.setMinimumWidth(120)

        self.control_panel = QVBoxLayout()
        self.control_panel.addWidget(self.game_status)
        self.control_panel.addWidget(self.reset_button)
        self.control_panel.addWidget(self.exit_button)

        self.main_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.main_layout.addWidget(self.board)
        self.main_layout.addLayout(self.control_panel)

        # Set layout
        self.setLayout(self.main_layout)

    def resizeEvent(self, event):
        offset = -self.board.main_layout.spacing() * self.board.main_layout.rowCount()
        width = self.size().width() + offset
        height = self.size().height() + offset
        excess = width - height
        if excess >= 0:
            min_cp_width = self.control_panel.minimumSize().width()
            margin = max([0, min_cp_width - excess + self.main_layout.spacing()])
            height -= margin
            self.main_layout.setDirection(QBoxLayout.Direction.LeftToRight)
            self.board.setFixedSize(height, height)
            self.board.setContentsMargins(0, 0, 0, 0)
        else:
            min_cp_height = self.control_panel.minimumSize().height()
            margin = max([0, min_cp_height + excess + self.main_layout.spacing()])
            width -= margin
            self.main_layout.setDirection(QBoxLayout.Direction.TopToBottom)
            self.board.setFixedSize(width + margin - self.main_layout.spacing(), width)
            self.board.setContentsMargins(
                (margin - self.main_layout.spacing()) / 2,
                0,
                (margin - self.main_layout.spacing()) / 2,
                0,
            )
        super().resizeEvent(event)

    def reset_game(self):
        for cell in self.board.findChildren(QPushButton):
            cell.setText(" ")
            cell.setStyleSheet("")
        self.game_status.setText("")

    def get_state(self):
        state = []
        cell_type = {"X": 1.0, "O": -1.0, " ": 0.0}
        for cell in self.board.findChildren(QPushButton):
            state.append(cell_type[cell.text()])
        # Add a batch dimension
        return torch.tensor(state).unsqueeze(0).to(self.device)

    def num_actions_available(self):
        num_actions_available = 0
        for cell in self.board.findChildren(QPushButton):
            if cell.text() == " ":
                num_actions_available += 1
        return num_actions_available

    def take_action(self, action):
        player = "X" if self.board.x_turn else "O"
        print(self.board.findChildren(QPushButton)[action.item()])
        self.board.findChildren(QPushButton)[action.item()].setText(player)
        self.board.x_turn = not self.board.x_turn
        if (game_status := self.board.check_game_status()) == 0:
            self.game_status.setText("It's a tie!")
            self.done = True
            return torch.tensor([1], device=self.device)
        elif game_status == 1:
            self.game_status.setText("X has won!")
            self.done = True
            return torch.tensor([-1 + (self.board.x_turn) * 3], device=self.device)
        elif game_status == -1:
            self.game_status.setText("O has won!")
            self.done = True
            return torch.tensor([2 + (self.board.x_turn) * -3], device=self.device)
        return torch.tensor([0], device=self.device)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Game screen
    game_screen = GameScreen()
    main_window = QMainWindow()
    main_window.setCentralWidget(game_screen)
    with open("stylesheet.qss", "r") as stylesheet:
        main_window.setStyleSheet(stylesheet.read())
    main_window.show()

    sys.exit(app.exec_())
