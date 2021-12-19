import chess, math, random, sys, os
from stockfish import Stockfish


"""
This is the driver method.

It takes the depth we want to explore, the state of the chess board, and a boolean isMaximizing, which represents whether we are currently evaluating a max or a min node

"""
PIECESCORE = {"p":1, "n":3, "b": 3, "r":5, "q":9, "k":900}
DEPTH = 3

def minimax_driver(depth, board, maximizing):
    # first we want to get a list of legal moves that we will iterate over, evaluate, and compare

    possibleMoves = list(board.legal_moves)
    #random.shuffle(possibleMoves)
    # we just want to set this to some arbitrarily large negative number in order to ensure the move we calculate is better
    bestMoveWeight = -9999
    bestMove = None

    #if len(possibleMoves) <= 1:
        #print(possibleMoves)
    
    for move in possibleMoves: 
        currentMove = chess.Move.from_uci(str(move))
        bestMove  = currentMove
        # apply the current move we are looking at
        board.push(currentMove)
        # here we want to select either our current bestMove, or the the move we are returning from minimax
        # i.e. if our bestMove is better than what we are getting from minimax keep it, else swap it
        weight = max(bestMoveWeight, minimax(depth - 1, board,-10000,10000, not maximizing))
        # print("current move:", move, "weight:", weight)
        #undo our applied move
        board.pop()

        if weight > bestMoveWeight: 
            bestMoveWeight = weight
            bestMove  = currentMove
    return bestMove

"""
This is the actual recursive method
"""
def minimax(depth, board, alpha, beta, maximizing):
    if depth == 0 :
        return -evaluation(board)
    possibleMoves = board.legal_moves
    if maximizing:
        bestMove = -9999
        for x in possibleMoves:
            move = chess.Move.from_uci(str(x))
            board.push(move)
            bestMove = max(bestMove,minimax(depth - 1, board,alpha,beta, not maximizing))
            board.pop()
            alpha = max(alpha,bestMove)
            if beta <= alpha:
                return bestMove
        return bestMove
    else:
        bestMove = 9999
        for x in possibleMoves:
            move = chess.Move.from_uci(str(x))
            board.push(move)
            bestMove = min(bestMove, minimax(depth - 1, board,alpha,beta, not maximizing))
            board.pop()
            beta = min(beta,bestMove)
            if beta <= alpha:
                return bestMove
        return bestMove
"""
This method takes the state of the board, and calculates the score using the FICS standard weight for each piece
"""
def evaluation(board):
    evaluation = 0
    isWhite = True

    for i in range(64):
        #print(evaluation)
        try:
            isWhite = bool(board.piece_at(i).color)
        except AttributeError as e:
            isWhite = isWhite
        pieceVal = 0 if str(board.piece_at(i)) == None else int(PIECESCORE.get(str(board.piece_at(i)).lower(), "0"))
        evaluation = evaluation + (pieceVal if isWhite else -pieceVal)
    return evaluation

def clear():
    os.system('cls')
def main():
    board = chess.Board()
    print(board)
    n = 0
    while n < 100:
        if n%2 == 0:
            move = input("Enter move: ")
            move = chess.Move.from_uci(str(move))
            board.push(move)
            clear()
            print(board)
        else:
            print("Computers Turn....")
            print("depth:", DEPTH)
            move = minimax_driver(DEPTH,board,True)
            move = chess.Move.from_uci(str(move))
            board.push(move)
            clear()
            print(board)
        n += 1
        

if __name__ == "__main__":
    main()