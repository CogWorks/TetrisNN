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
        self.actions = 5
        self.finished = False
        self.game = 0
        self.level = 0
        self.lines = 0
        self.score = 0
        self.reward = 0
        self.frames = 0
        self.episodes = 0
        self.best_score = 0
        self.best_episodes = 0
        self.best_lines = 0
        self.log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"deepqlearning.log"),"w")
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
        
    def get_points(self, N):
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
        points = 0
        reward = -0.1
        zoid = all_zoids[self.zoid_name].get_copy()
        temp_board = self.board.get_cow()
        zoid.set_orient(self.zoid_orient)
        if index==2:
            reward = 0
            while True:
                if self.zoid_row > 0:
                    if temp_board.imprint_zoid(zoid, pos=(self.zoid_row-1, self.zoid_col), value=1, check=True):
                        self.zoid_board = tetris_cpp.tetris_cow2()
                        self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                        self.zoid_row = self.zoid_row-1
                        zoid = all_zoids[self.zoid_name].get_copy()
                        temp_board = self.board.get_cow()
                        zoid.set_orient(self.zoid_orient)
                        points += 1
                    else:
                        break
                else:
                    break
            points += self.new_zoid(True)
        elif index==4:
            if self.zoid_row > 0:
                if temp_board.imprint_zoid(zoid, pos=(self.zoid_row-1, self.zoid_col), value=1, check=True):
                    self.zoid_board = tetris_cpp.tetris_cow2()
                    self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                    self.zoid_row = self.zoid_row-1
                    reward = 0
        elif index==0:
            if self.zoid_col > 0:
                if temp_board.imprint_zoid(zoid, pos=(self.zoid_row, self.zoid_col-1), value=1, check=True):
                    self.zoid_board = tetris_cpp.tetris_cow2()
                    self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                    self.zoid_col = self.zoid_col-1
                    reward = 0
        elif index==1:
            if self.zoid_col < 9-zoid.col_count():
                if temp_board.imprint_zoid(zoid, pos=(self.zoid_row, self.zoid_col+1), value=1, check=True):
                    self.zoid_board = tetris_cpp.tetris_cow2()
                    self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                    self.zoid_col = self.zoid_col+1
                    reward = 0
        elif index==3:
            if self.zoid_name == "I" and self.zoid_row == 19 and self.zoid_orient == 0:
                self.zoid_row = 18
            zoid.set_orient((self.zoid_orient+1)%4)
            if self.zoid_col+zoid.col_count() <= 10:
                if temp_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1, check=True):
                    self.zoid_board = tetris_cpp.tetris_cow2()
                    self.zoid_board.imprint_zoid(zoid, pos=(self.zoid_row,self.zoid_col), value=1)
                    self.zoid_orient = zoid.get_orient()
                    reward = 0
        self._show()
        self.score += points
        r = points + reward
        self.reward += r
        return r

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
        print("Game: %d" % self.game)
        print("Episode: %d" % self.episodes)
        print("Score: %d" % self.score)
        print("Lines: %d" % self.lines)
        print("Best Episode: %d" % self.best_episodes)
        print("Best Score: %d" % self.best_score)
        print("Best Lines: %d" % self.best_lines)
        
    def new_zoid(self, oldzoid=None):
        points = 0
        if oldzoid:
            zoid = all_zoids[self.zoid_name].get_copy()
            zoid.set_orient(self.zoid_orient)
            self.board.imprint_zoid(zoid, pos=(self.zoid_row, self.zoid_col), value=1, check=True)
            lines = self.board.check_full(False)
            points += self.get_points(lines)
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
        self.frames = 0
        return points

    def reset(self):
        if self.game > 0:
            self.log.write("%d,%d,%d,%d,%d\n" % (self.game, self.episodes, self.lines, self.score, self.reward))
            self.log.flush()
        if self.episodes > self.best_episodes:
            self.best_episodes = self.episodes
        if self.lines > self.best_lines:
            self.best_lines = self.lines
        if self.score > self.best_score:
            self.best_score = self.score
        self.level = 0
        self.lines = 0
        self.reward = 0
        self.score = 0
        self.frames = 0
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
    sgd = PyDeepCL.SGD(cl, 0.05, 0)
    sgd.setMomentum(0.0001)
    net.addLayer(PyDeepCL.InputLayerMaker().numPlanes(planes).imageSize(size))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(10).filterSize(5).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(PyDeepCL.PoolingMaker().poolingSize(3))
    net.addLayer(PyDeepCL.ConvolutionalMaker().numFilters(10).filterSize(5).padZeros().biased())
    net.addLayer(PyDeepCL.ActivationMaker().relu())
    net.addLayer(PyDeepCL.PoolingMaker().poolingSize(5))
    net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(200).imageSize(1).biased())
    net.addLayer(PyDeepCL.ActivationMaker().tanh())
    net.addLayer(PyDeepCL.FullyConnectedMaker().numPlanes(7).imageSize(1).biased())
    net.addLayer(PyDeepCL.SoftMaxMaker())

    simulator.setNet(net)

    qlearner = PyDeepCL.QLearner(sgd, simulator, net)
    qlearner.setLambda(0.9) # sets decay of the eligibility trace decay rate
    qlearner.setMaxSamples(32) # how many samples to learn from after each move
    qlearner.setEpsilon(0.1) # probability of exploring, instead of exploiting
    #qlearner.setLearningRate(0.1)
    qlearner.run()

if __name__ == '__main__':
    go()


