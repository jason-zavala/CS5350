from stockfish import Stockfish
import miniMaxChess, chess
from miniMaxChess import clear, minimax_driver


stockfish = Stockfish("C:\\Users\\Jason Zavala\\Documents\School\\repo\CS5350\\Final Project\\stockfish_14.1_win_x64_avx2.exe")
DEPTH = 3

def main():
    board = chess.Board()
    print(board)
    print()
    n = 0
    while n < 100:
        if n%2 == 0:
            move = str(stockfish.get_best_move())
            stockfish.make_moves_from_current_position([move])
            print("Stockfish moves:", move)
            move = chess.Move.from_uci(str(move))
            
            board.push(move)
            print(board)
            print()
        else:
            print("LiteBlue's turn....")
            print("depth:", DEPTH, "N: ", n)
            # if n > 30: 
            #     DEPTH = 5
            print()
            move = minimax_driver(DEPTH,board,True)
            print("LiteBlue moves:", move)
            stockfish.make_moves_from_current_position([move])
            move = chess.Move.from_uci(str(move))
            board.push(move)
            print(board)
            print()
        if board.is_checkmate():
            print("CHECKMATE")
            break
        n += 1

        

if __name__ == "__main__":
    main()