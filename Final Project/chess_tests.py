from os import times
from stockfish import Stockfish
import miniMaxChess, chess
import time
from miniMaxChess import clear, minimax_driver

stockfish = Stockfish("stockfish_14.1_win_x64_avx2.exe")

def run_game(depth):
    board = chess.Board()
    stockfish.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    #print(board)
    #print()
    n = 0
    while n < 100:
        if n%2 == 0:
            move = str(stockfish.get_best_move())
            stockfish.make_moves_from_current_position([move])
            #print("Stockfish moves:", move)
            move = chess.Move.from_uci(str(move))
            
            board.push(move)
            #print(board)
            #print()
        else:
            #print("LiteBlue's turn....")
            #print("depth:", depth, "N: ", n)
            # if n > 30: 
            #     DEPTH = 5
            #print()
            move = minimax_driver(depth,board,True)
            if move is None:
                continue
            #print("LiteBlue moves:", move)
            stockfish.make_moves_from_current_position([move])
            move = chess.Move.from_uci(str(move))
            board.push(move)
            #print(board)
            #print()
        n += 1
        if board.is_checkmate():
            if board.outcome().result() is None:
                return('1/2-1/2', n)
            return(board.outcome().result(), n)
        

#DEPTH = 3

def main():
    depth_range = [1, 2, 3]
    stockfish.set_depth(1)
    stockfish.set_elo_rating(0)
    epoch = 10

    for depth in depth_range:
        print("running depth: ", depth)
        results = []
        start_time = time.time()
        for t in range(epoch):
            print("\t running epoch: ", t)
            results.append(run_game(depth))
        elapsed_time = round(time.time() - start_time, 4)

        print()
        print("results", results)
        percent_white_won = [r[0] for r in results].count('1-0') / len(results)
        print("percentage stockfish won:", percent_white_won)
        avg_num_moves = sum([r[1] for r in results]) / len(results)
        print("average game length (moves): ", avg_num_moves)
        print(f"time to run {epoch} games with depth {depth}: {elapsed_time}")
        

if __name__ == "__main__":
    main()