#!/usr/bin/env python

from __future__ import print_function

import sys
import array
import random
import random
import PyDeepCL

import tetris_cpp
from boards._helpers import print_board
from boards.tetris_zoid import *

def copy_board_zoid(board, *args, **kwds):
    zoid_board = board.get_cow()
    zoid_board.imprint_zoid(*args, check=True, **kwds)
    return zoid_board

def copy_board_hide(board):
    cleared_board = board.get_cow()
    cleared_board.check_full(True)
    return cleared_board

class TetrisSimulator(PyDeepCL.Scenario):

    def __init__(self):
        super(TetrisSimulator, self).__init__()
        self.size = 20
        self.planes = 2
        self.actions = 3
        self.finished = False
        self.game = 0
        self.reset()

    def getPerceptionSize(self):
        return self.size
        
    def getNumActions(self):
        return self.actions
        
    def getPerceptionPlanes(self):
        return self.planes
        
    def getPerception(self):
        perception = [0] * self.planes * self.size * self.size
        if self.board.pile_height() > 0:
            for r in self.board.rows():
                for c in self.board.cols():
                    if self.board[r,c] > 0:
                        perception[r * self.size + c] = 1;
        # perception[self.size * self.size + r * self.size + c] = 1;
        return perception
        
    def act(self,index):
        reward = 0
        zoid_row = self.zoid_row
        zoid_column = self.zoid_column
        zoid_orient = self.zoid_orient
        if index==0:
            if self.zoid_column + self.zoid.col_count() < 10:
                zoid_column += 1
            else:
                reward = -1
        elif index==1:
            if self.zoid_column > 1:
                zoid_column -= 1
            else:
                reward = -1
        elif index==2:
            if self.zoid_row > 0:
                zoid_row -= 1
            elif self.zoid_row == 0:
                self.finished = True
        zoid_board = copy_board_zoid(self.board, self.zoid, pos=(zoid_row, zoid_column))
        self.zoid_board = zoid_board
        self.zoid_row = zoid_row
        self.zoid_column = zoid_column
        self.zoid_orient = zoid_orient
        self._show()
        return reward

    def hasFinished(self):
        return self.finished

    def setNet(self, net):
        self.net = net
        
    def _show(self):
        print_board(self.zoid_board, entire=True)
            
    def reset(self):
        self.board = tetris_cpp.tetris_cow2()
        self.zoid_name, self.zoid = random.choice(all_zoids.items())
        self.zoid = self.zoid.get_copy()
        self.zoid_orient = 0
        self.zoid_profile = self.zoid.get_bottom_profile()
        self.zoid_column = int(round((10-len(self.zoid_profile))/2.0))
        self.zoid_row = 20 - self.zoid.row_count()
        self.zoid_board = tetris_cpp.tetris_cow2()
        self.zoid_board.imprint_zoid(self.zoid, orient=0, pos=(self.zoid_row, self.zoid_column), check=True)
        #if self.game > 0:
        self._show()
        self.finished = False
        self.game += 1

def go():
    simulator = TetrisSimulator()

    size = simulator.getPerceptionSize();
    planes = simulator.getPerceptionPlanes();
    numActions = simulator.getNumActions();

    cl = PyDeepCL.EasyCL()
    net = PyDeepCL.NeuralNet(cl)
    sgd = PyDeepCL.SGD(cl, 0.1, 0.0)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(8).filterSize(5).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(100).imageSize(1).biased())
    net.addLayer(PyDeepCL.ActivationMaker().tanh())
    net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(numActions).imageSize(1).biased())
    net.addLayer(PyDeepCL.SquareLossMaker())

    simulator.setNet(net)

    qlearner = PyDeepCL.QLearner(sgd, simulator, net)
    qlearner.run()

if __name__ == '__main__':
    go()


