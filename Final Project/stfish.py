from stockfish import Stockfish

stockfish = Stockfish("C:\\Users\\Jason Zavala\\Documents\School\\repo\CS5350\\Final Project\\stockfish_14.1_win_x64_avx2.exe")

def print_board():
    print( stockfish.get_board_visual() )

def main():
    board = stockfish.get_board_visual()
    mv = stockfish.get_best_move()
    stockfish.make_moves_from_current_position([mv])
    print_board()

    mv = stockfish.get_best_move()
    print(mv)
    stockfish.make_moves_from_current_position([mv])
    print_board()


if __name__ == "__main__":
    main()