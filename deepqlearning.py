#!/usr/bin/env python

from __future__ import print_function

import sys, os
import array
import random
import time
import PyDeepCL

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"tetris_cpp"))
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
        self.actions = 4
        self.finished = False
        self.game = 0
        for zoid_name,zoid in all_zoids.items():
            for orient in xrange(4):
                zoid = zoid.get_copy()
                zoid.set_orient(orient)
                print_board(zoid)
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
            for r in range(0,20):
                for c in range(0,10):
                    if self.board[21-r,c] > 0:
                        perception[r * self.size + c] = 1;
        for r in range(0,20):
            for c in range(0,10):
                if self.zoid_board[21-r,c] > 0:
                    perception[self.size * self.size + r * self.size + c] = 1;
        return perception
        
    def get_reward(self, N):
        if N==1:
            return 40 * (self.level + 1)
        elif N==2:
            return 100 * (self.level + 1)
        elif N==3:
            return 300 * (self.level + 1)
        elif N==4:
            return 1200 * (self.level + 1)
        else:
            return -0

    def act(self,index):
        print("Index: %d" % index)
        reward = 1
        zoid = all_zoids[self.zoid_name].get_copy()
        temp_board = self.board.get_cow()
        zoid.set_orient(self.zoid_orient)
        if index==3:
            if self.zoid_row > 0:
                if temp_board.imprint_zoid(zoid, pos=(self.zoid_row-1, self.zoid_col), value=1, check=True):
                    self.zoid_board = tetris_cpp.tetris_cow2()
                    self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                    self.zoid_row = self.zoid_row-1
                else:
                    reward += self.new_zoid(True)
            else:
                reward += self.new_zoid(True)
        elif index==0:
            if self.zoid_name == "I" and self.zoid_row == 19 and self.zoid_orient == 0:
                self.zoid_row = 18
            zoid.set_orient((self.zoid_orient+1)%4)
            if self.zoid_col+zoid.col_count() <= 10 and temp_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1, check=True):
                self.zoid_board = tetris_cpp.tetris_cow2()
                self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                self.zoid_orient = zoid.get_orient()
        elif index==1:
            if self.zoid_col > 0:
                if temp_board.imprint_zoid(zoid, pos=(self.zoid_row, self.zoid_col-1), value=1, check=True):
                    self.zoid_board = tetris_cpp.tetris_cow2()
                    self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                    self.zoid_col = self.zoid_col-1
        elif index==2:
            if self.zoid_col < 9-zoid.col_count():
                if temp_board.imprint_zoid(zoid, pos=(self.zoid_row, self.zoid_col+1), value=1, check=True):
                    self.zoid_board = tetris_cpp.tetris_cow2()
                    self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                    self.zoid_col = self.zoid_col+1
        self._show()
        return reward

    def hasFinished(self):
        return self.finished

    def setNet(self, net):
        self.net = net
        
    def _show(self):
        zoid = all_zoids[self.zoid_name].get_copy()
        zoid.set_orient(self.zoid_orient)
        combo_board = self.board.get_cow()
        combo_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=2)
        print_board(combo_board, count=20)
        print("Episode: %d" % self.episodes)
        print("Score: %d" % self.score)
        print("Lines: %d" % self.lines)
        
    def new_zoid(self, oldzoid=None):
        reward = 0
        if oldzoid:
            zoid = all_zoids[self.zoid_name].get_copy()
            zoid.set_orient(self.zoid_orient)
            self.board.imprint_zoid(zoid, pos=(self.zoid_row, self.zoid_col), value=1, check=True)
            lines = self.board.check_full(False)
            reward += self.get_reward(lines)
            self.score += reward
            self.lines += lines
        self.episodes += 1
        self.zoid_board = tetris_cpp.tetris_cow2()
        self.zoid_name = random.choice(all_zoids.keys())
        self.zoid_orient = 0
        self.zoid_col = 3
        if self.zoid_name == "I":
            self.zoid_row = 18
        else:
            self.zoid_row = 18
            if self.zoid_name == "O":
                self.zoid_col = 4
        zoid = all_zoids[self.zoid_name].get_copy()
        zoid.set_orient(self.zoid_orient)
        temp_board = self.board.get_cow()
        if temp_board.imprint_zoid(zoid, pos=(self.zoid_row, self.zoid_col), value=1, check=True):
            self.zoid_board = temp_board
        else:
            self.finished = True
            #reward += self.episodes
        return reward

    def reset(self):
        self.level = 0
        self.lines = 0
        self.score = 0
        self.episodes = 0
        self.game += 1
        self.board = tetris_cpp.tetris_cow2()
        self.new_zoid()
        self._show()
        self.finished = False

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


