from src.Board import Board
from src.AlphaZero import AlphaZero

PLAYER1 = 1
PLAYER2 = -1

alphaZero = AlphaZero()
alphaZero.load_network()

board = Board()
alphaZero_choise = alphaZero.play_with_human(board.board, board.last_position)
print("alpha zero's choise is: ", alphaZero_choise)
board.place(alphaZero_choise, -1)
print(board.see_board)

while board.check_continue():
    player1_choise = (input("Wow, which sub-board would you like to play on"))
    player1_choise = list(map(int, player1_choise.split(",")))
    player1_choise = tuple(player1_choise)
    board.place(player1_choise, 1)
    print(board.see_board)
    if board.get_winner() == 1:
        print("human wins")
        break
    alphaZero_choise = alphaZero.play_with_human(board.board, board.last_position)
    print("alpha zero's choise is: ", alphaZero_choise)

    board.place(alphaZero_choise, -1)
    print(board.see_board)

print("alpha zero wins")