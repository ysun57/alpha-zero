# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from src.Board import Board
from src.Agent import Agent

import random
import time
import os

PLAYER1 = 1
PLAYER2 = -1

history = []

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    player1 = Agent(player=PLAYER1)  # player 1 plays X's
    player2 = Agent(player=PLAYER2)  # player 2 plays O's
    players = [player1, player2]
    wins = []  # [player1 wins, player2 wins, draws]

    board = Board()

    n_games = 10000
    print("start")
    t = str(time.asctime(time.localtime(time.time())))
    t = t.replace(" ", "_").replace(":", "-")
    os.makedirs(t, exist_ok=True)
    for game in range(n_games):
        board.reset()  # reset board
        board.place(player1.select_action(board.board, board.check_valid_position()), PLAYER1)
        while board.check_continue():
            board.place(player2.select_action(board.board, board.check_valid_position()), PLAYER2)
            if not board.check_continue():
                break
            board.place(player1.select_action(board.board, board.check_valid_position()), PLAYER1)
        winner = board.winner  # player1 win: 1, draw: 0, player2 win: -1
        [player.learn(winner) for player in players]  # learn from experiences for both players
        [player.update_epsilon() for player in players]  # update epsilon for both agents

        wins.append(winner)

        if game % 10 == 0 and game != 0:
            player1.save(game, t)
            player2.save(game, t)
            print("Played " + str(game) + " games")
            print("X wins for last {} games are {}%".format(game, (wins[-10000:].count(1) / game) * 100))
            print("O wins for last {} games are {}%".format(game, (wins[-10000:].count(-1) / game) * 100))
            print("Draws for last {} games are {}%".format(game, (wins[-10000:].count(0) / game) * 100))
            print("Player epsilon is {}".format(player1.epsilon))
            history.append(player1.epsilon)
            print(history)
            print("")

        if game > 990000:
            print(board)
            print("winner is " + str(wins[-1]))
            print("")
