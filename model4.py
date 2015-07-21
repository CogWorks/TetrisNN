#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI_2"))

import math, random

import simulator
import tetris_cpp
from boards._helpers import print_board
from boards.tetris_zoid import *

    
def row_sum(row, board):
    return sum([0 if c==0 else 1 for c in board.row_iter(row)])
    
def board_sum(board):
    return sum([row_sum(r, board) for r in board.rows()])
    
def copy_board_zoid(board,*args,**kwds):
    zoid_board = board.get_cow()
    zoid_board.imprint_zoid(*args,check=True,**kwds)
    return zoid_board

def copy_board_hide(board):
    cleared_board = board.get_cow()
    cleared_board.check_full(True)
    return cleared_board
    
def fullLines(board):
    return any([row_sum(r, board)>9 for r in board.rows()])
    
def noIllegal(board):
    for r in board.rows():
        for c in board.cols():
            s = 0
            if c!=0 and board[r,c-1]!=0:
                s += 1
            if c!=9 and board[r,c+1]!=0:
                s += 1
            if r!=0 and board[r-1,c]!=0:
                s += 1
            if r!=19 and board[r+1,c]!=0:
                s += 1
            if s==0:
                return False
    return True
    
if __name__ == '__main__':
    max_rows = 10
    main_board = tetris_cpp.tetris_cow2(rows=max_rows)
    for row in main_board.rows():
        for col in main_board.cols():
            main_board[row,col] = 1
    while fullLines(main_board):
        new_board = main_board.get_cow()
        new_board[random.choice(range(0,max_rows)),random.choice(range(0,10))] = 0
        if noIllegal(new_board):
            main_board = new_board
    print_board(main_board,entire=True)