#!/usr/bin/env python

import sys, os, termios
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI_2"))

import tetris_cpp
from boards._helpers import print_board

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

import cPickle

import numpy as np

TERMIOS = termios
def getkey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~TERMIOS.ICANON & ~TERMIOS.ECHO
    new[6][TERMIOS.VMIN] = 1
    new[6][TERMIOS.VTIME] = 0
    termios.tcsetattr(fd, TERMIOS.TCSANOW, new)
    c = None
    try:
        c = os.read(fd, 1)
    finally:
        termios.tcsetattr(fd, TERMIOS.TCSAFLUSH, old)
    return c

def null_board():
    return np.zeros(200).reshape(20,10)
    
pieces = ["I", "O", "T", "S", "Z", "J", "L"]
def to_one_hot(value, labels):
    return [1 if value==l else 0 for l in labels]

boards = []
zoids = []
actions = []

# Perfect I line clears (orient 0)
for cc in range(0,7):
    boards.append(null_board())
    zoids.append("I")
    actions.append(10*0+cc)
    for c in range(0,10):
        for r in range(0,1):
            if c < cc or c >= cc+4:
                boards[-1][19-r][c] = 1

# Perfect I line clears (orient 1)
for cc in range(0,10):
    boards.append(null_board())
    zoids.append("I")
    actions.append(10*1+cc)
    for c in range(0,10):
        for r in range(0,4):
            if c!=cc:
                boards[-1][19-r][c] = 1

# Perfect 0 lines clears (orient 0)
for cc in range(0,9):
    boards.append(null_board())
    zoids.append("O")
    actions.append(10*0+cc)
    for c in range(0,10):
        for r in range(0,2):
            if c < cc or c >= cc+2:
                boards[-1][19-r][c] = 1

# Perfect T lines clears (orient 0) nub down
for cc in range(0,8):
    boards.append(null_board())
    zoids.append("T")
    actions.append(10*0+cc)
    for c in range(0,10):
        if c < cc or c >= cc+3:
            boards[-1][18][c] = 1
        if c <= cc or c >= cc+2:
            boards[-1][19][c] = 1
            
# Perfect T lines clears (orient 1) nub left
for cc in range(0,9):
    boards.append(null_board())
    zoids.append("T")
    actions.append(10*1+cc)
    for c in range(0,10):
        if c < cc or c >= cc+2:
            boards[-1][18][c] = 1
        if c != cc+1:
            boards[-1][19][c] = 1

# Imperfect T lines clears (orient 2) nub up
for cc in range(0,8):
    boards.append(null_board())
    zoids.append("T")
    actions.append(10*2+cc)
    for c in range(0,10):
        for r in range(0,1):
            if c < cc or c >= cc+3:
                boards[-1][19-r][c] = 1

# Perfect T lines clears (orient 3) nub right
for cc in range(0,9):
    boards.append(null_board())
    zoids.append("T")
    actions.append(10*3+cc)
    for c in range(0,10):
        if c < cc or c > cc+1:
            boards[-1][18][c] = 1
        if c < cc or c > cc:
            boards[-1][19][c] = 1
            
# Imperfect L lines clears (orient 0) nub down
for cc in range(0,8):
    boards.append(null_board())
    zoids.append("L")
    actions.append(10*0+cc)
    for c in range(0,10):
        if c < cc or c > cc+2:
            boards[-1][18][c] = 1
        if c < cc or c > cc:
            boards[-1][19][c] = 1
            
# Perfect L lines clears (orient 1) nub left
for cc in range(0,9):
    boards.append(null_board())
    zoids.append("L")
    actions.append(10*1+cc)
    for c in range(0,10):
        if c < cc or c > cc+1:
            boards[-1][17][c] = 1
        if c <= cc or c > cc+1:
            boards[-1][18][c] = 1
        if c <= cc or c > cc+1:
            boards[-1][19][c] = 1
            
# Imperfect L lines clears (orient 2) nub up
for cc in range(0,8):
    boards.append(null_board())
    zoids.append("L")
    actions.append(10*2+cc)
    for c in range(0,10):
        if c < cc or c > cc+2:
            boards[-1][19][c] = 1
            
# Imperfect J lines clears (orient 0) nub down
for cc in range(0,8):
    boards.append(null_board())
    zoids.append("J")
    actions.append(10*0+cc)
    for c in range(0,10):
        if c < cc or c > cc+2:
            boards[-1][18][c] = 1
        if c <= cc+1 or c > cc+2:
            boards[-1][19][c] = 1
                        
# Imperfect J lines clears (orient 2) nub up
for cc in range(0,8):
    boards.append(null_board())
    zoids.append("J")
    actions.append(10*2+cc)
    for c in range(0,10):
        if c < cc or c > cc+2:
            boards[-1][19][c] = 1

# Perfect J lines clears (orient 3) nub right
for cc in range(0,9):
    boards.append(null_board())
    zoids.append("J")
    actions.append(10*3+cc)
    for c in range(0,10):
        if c < cc or c > cc+1:
            boards[-1][17][c] = 1
        if c < cc or c > cc:
            boards[-1][18][c] = 1
        if c < cc or c > cc:
            boards[-1][19][c] = 1
            
# Imperfect Z lines clears (orient 0)
for cc in range(1,9):
    boards.append(null_board())
    zoids.append("Z")
    actions.append(10*0+cc)
    for c in range(0,10):
        if c < cc or c > cc+1:
            boards[-1][19][c] = 1

# Imperfect Z lines clears (orient 1)
for cc in range(0,9):
    boards.append(null_board())
    zoids.append("Z")
    actions.append(10*1+cc)
    for c in range(0,10):
        if c < cc or c > cc+1:
            boards[-1][18][c] = 1
        if c < cc or c > cc:
            boards[-1][19][c] = 1
            
# Imperfect S lines clears (orient 0)
for cc in range(0,8):
    boards.append(null_board())
    zoids.append("S")
    actions.append(10*0+cc)
    for c in range(0,10):
        if c < cc or c > cc+1:
            boards[-1][19][c] = 1
            
# Imperfect S lines clears (orient 1)
for cc in range(0,9):
    boards.append(null_board())
    zoids.append("S")
    actions.append(10*1+cc)
    for c in range(0,10):
        if c < cc or c > cc+1:
            boards[-1][18][c] = 1
        if c < cc+1 or c > cc+1:
            boards[-1][19][c] = 1
            
# # Empty boards I
# boards.append(null_board())
# zoids.append("I")
# actions.append(10*1+0)
# boards.append(null_board())
# zoids.append("I")
# actions.append(10*1+9)
#
# # Empty boards O
# boards.append(null_board())
# zoids.append("I")
# actions.append(10*0+0)
# boards.append(null_board())
# zoids.append("I")
# actions.append(10*0+8)
#
# # Empty boards L
# boards.append(null_board())
# zoids.append("L")
# actions.append(10*3+0)
# boards.append(null_board())
# zoids.append("L")
# actions.append(10*2+8)
#
# # Empty boards J
# boards.append(null_board())
# zoids.append("J")
# actions.append(10*2+0)
# boards.append(null_board())
# zoids.append("J")
# actions.append(10*1+8)

nboards = len(boards)
# inputs = []
# outputs = []
for i in range(0,nboards):
    print ",".join(map(str,[int(boards[i][r,c]) for r in range(0,20) for c in range(0,10)] + to_one_hot(zoids[i], pieces) + [actions[i]]))
    # inputs.append([int(boards[i][r,c]) for r in range(0,20) for c in range(0,10)] + to_one_hot(zoids[i], pieces))
    # outputs.append()
    #board = tetris_cpp.tetris_cow2.convert_old_board(boards[i])
    #print_board(board)
    # getkey()
#print(nboards)
# with open("master_training_data.dat", "w") as f:
#     f.write(cPickle.dumps(DenseDesignMatrix(X=np.array(inputs), y=np.array(outputs), y_labels = 40)))