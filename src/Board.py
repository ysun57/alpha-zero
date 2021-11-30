# this is the game board for Ultimate tic tac toe game

import numpy as np
import random


class Board:
    scaler = 0.1
    board = np.ones((9, 9)) * scaler
    player1 = "x"
    player2 = "o"
    empty_place = "-"
    count = 0
    last_position = 10
    prohibited_area = []
    winner = 0

    def __init__(self):
        self.board = np.ones((9, 9)) * self.scaler
        self.count = 0

    def reset(self):
        self.board = np.ones((9, 9)) * self.scaler
        self.count = 0
        self.last_position = 10
        self.prohibited_area = []
        self.winner = 0

    def place(self, position, player):
        self.board[position] = player
        self.last_position = position[1]

    def check_valid_position(self):
        def create_tuple(x):
            target = []
            if type(x) == tuple:
                for i in range(len(x[0])):
                    if i not in self.prohibited_area:
                        target.append((x[0][i], x[1][i]))
            elif len(x.shape) == 1:
                for i in range(len(x)):
                    target.append((self.last_position, x[i]))
            return target


        if self.last_position < 9:
            pos = np.where(self.board[self.last_position, :] == self.scaler)[0]
            if len(pos) > 0:
                return create_tuple(pos)
            else:
                return create_tuple(np.where(self.board == self.scaler))
            # return self.board[self.last_position, :][self.board[self.last_position, :] != 0]
        else:
            return create_tuple(np.where(self.board == self.scaler))

    def check_continue(self):
        winner = self.get_winner()
        if winner != 0:
            self.winner = winner
            return False;
        if self.count == 81:
            return False
        if not self.check_valid_position():
            return False
        return True

    @property
    def see_board(self):

        def pixel_board(x):
            if x == 1:
                return self.player1
            elif x == -1:
                return self.player2
            else:
                return self.empty_place

        def list_board(x):
            return list(map(pixel_board, x))

        visual_board = ''

        for i in range(3):
            for j in range(3):
                visual_board = visual_board + ' '.join(list_board(self.board[3 * i, 3 * j: 3 * j + 3])) + ('  |  ')
                visual_board = visual_board + ' '.join(list_board(self.board[3 * i + 1, 3 * j: 3 * j + 3])) + ('  |  ')
                visual_board = visual_board + ' '.join(list_board(self.board[3 * i + 2, 3 * j: 3 * j + 3])) + ('  |  ')
                visual_board = visual_board + "\n"
            visual_board = visual_board + "\n"

        return (visual_board)

    def get_winner(self):

        def get_part_winner(x):  # since these columns need to go through pos 4, can remove != 0 check for each
            if x[4] != self.scaler:
                if x[0] == x[4] == x[8]:  # Diagonal
                    return x[4]
                elif x[2] == x[4] == x[6]:  # Diagonal
                    return x[4]
                elif x[3] == x[4] == x[5]:  # Row
                    return x[4]
                elif x[1] == x[4] == x[7]:  # Column
                    return x[4]

            if x[0] != self.scaler:
                if x[0] == x[1] == x[2]:  # Row
                    return x[0]
                elif x[0] == x[3] == x[4]:  # Column
                    return x[0]

            if x[8] != self.scaler:
                if x[6] == x[7] == x[8]:  # Row
                    return x[8]
                elif x[2] == x[5] == x[8]:  # Column
                    return x[8]
            return 0

        win_map = []

        for i in range(len(self.board)):
            part_winner = get_part_winner(self.board[i])
            win_map.append(part_winner)
            if part_winner != 0:
                if i not in self.prohibited_area:
                    self.prohibited_area.append(i)
                    self.board[i] = np.where(self.board[i] != 0.1, self.board[i], 0)

        return get_part_winner(np.array(win_map))


def test():
    board = Board()
    for i in range(1000):
        board.reset()
        print(board.check_continue())
        player = 1
        board.place((4, 4), player)
        while board.check_continue():
            player = player * -1
            current_pos = (random.choice(board.check_valid_position()))
            board.place(current_pos, player)
        print(board.winner)
        print(board.prohibited_area)
        print(board.see_board)
        # print(board.board)
    print("test finish!")


if __name__ == '__main__':
    test()
