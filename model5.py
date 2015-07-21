#!/usr/bin/env python

"""
Artificial Neural Network Tetris Modeling
1. Generate dataset
1.1. Load boards.csv.gz
1.2. For each board pick random zoid and find the best place to put the zoid based on some score/cost
1.3. Create an state space vector from board layout and extra features (eg. current zoid)
2. Train ANN using new dataset
3. Evaluate ANN in Tetris Simulator
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI_2"))

import numpy as np
import gzip
import random

import tetris_cpp
from boards._helpers import print_board
from boards.tetris_zoid import *

from pylearn2.models import mlp
from pylearn2.train import Train
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import ChannelTarget, EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from multiprocessing import Pool
import cPickle

pieces = ["I", "O", "T", "S", "Z", "J", "L"]
def to_one_hot(value, labels):
    return [1 if value==l else 0 for l in labels]
    
def copy_board_zoid(board,*args,**kwds):
    zoid_board = board.get_cow()
    zoid_board.imprint_zoid(*args,check=True,**kwds)
    return zoid_board

def copy_board_hide(board):
    cleared_board = board.get_cow()
    cleared_board.check_full(True)
    return cleared_board

def score_board(board):
    if board.pile_height() == 0:
        return 1.0
    heights = [board.pile_height()-p for p in board.get_top_profile()]
    col_sums = [sum([0 if board[r,c]==0 else 1 for r in board.rows()]) for c in board.cols()]
    penalty = [heights[c]-col_sums[c] for c in board.cols()]
    w = [i for i in range(0,len(heights)) if heights[i]>0]
    if len(w)==0:
        print_board(board)
        print heights
        print col_sums
    ii = min(w)
    jj = max(w)
    width = jj - ii + 1
    height = max(heights)
    s = sum(col_sums)
    return 1.0*(s-sum(penalty))/(width*height)

def find_best_move(board, zoid_name):
    main_board = tetris_cpp.tetris_cow2.convert_old_board(board)
    oldscore = score_board(main_board)
    board_profile = main_board.get_top_profile()
    zoid = all_zoids[zoid_name].get_copy()
    scores = []
    for orient in xrange(4):
        zoid_profile = zoid.get_bottom_profile()
        for c in main_board.cols():
            if c+zoid.col_count() > main_board.col_count(): break
            heights = tuple(board_profile[cc+c]+zoid_profile[cc] for cc in xrange(len(zoid_profile)))
            r = main_board.pile_height()-min(heights)
            if r+zoid.row_count() > main_board.row_count(): continue
            zoid_board = copy_board_zoid(main_board,zoid,pos=(r,c),value=2)
            cleared_board = copy_board_hide(zoid_board)
            newscore = score_board(cleared_board)
            scores.append((newscore-oldscore,orient,(r,c),cleared_board))
    if not scores: return None
    scores.sort(key=lambda x: x[0],reverse=True)
    return scores[0]

def process_line(line):
    board = map(int,line.split(','))
    z = random.choice(pieces)
    zoid = to_one_hot(z, pieces)
    nb = np.array(board).reshape(20,10)
    bm = find_best_move(nb, z)
    if bm == None:
        return None
    else:
        return (board+zoid,[bm[1]*10+bm[2][1]])

def generate_training_data():
    with gzip.open('boards.csv.gz','r') as f:
        inputs = []
        targets = []
        lines = f.readlines()
        p = Pool(8)
        return p.map(process_line, lines)
        
def get_features(board, zoid):
    b = tetris_cpp.tetris_cow2.convert_old_board(np.array(board).reshape(20,10))
    profile = list(b.get_top_profile())
    heights = [b.pile_height()-p for p in profile]
    col_sums = [sum([0 if b[r,c]==0 else 1 for r in b.rows()]) for c in b.cols()]
    penalty = [heights[c]-col_sums[c] for c in b.cols()]
    density = 1
    w = [i for i in range(0,len(heights)) if heights[i]>0]
    if len(w) > 0:
        ii = min(w)
        jj = max(w)
        width = jj - ii + 1
        height = max(heights)
        s = sum(col_sums)
        density = 1.0*(s-sum(penalty))/(width*height)
    return board + zoid
    
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

class MyCostSubclass(DefaultDataSpecsMixin, Cost):
    
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        
        inputs, targets = data
        outputs = model.logistic_regression(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()
    
if __name__ == "__main__":
    
    # dm = generate_training_data()
    # with open("training_data.dat", "w") as f:
    #     f.write(cPickle.dumps(dm))
    with open("master_training_data.dat","r") as f:
        results = f.readlines()
        inputs = []
        targets = []
        for result in results:
            result = map(int,result.split(","))
            board = result[0:200]
            zoid = result[200:207]
            action = result[207]
            features = get_features(board,zoid)
            inputs.append(features)
            targets.append([action])
        data = DenseDesignMatrix(X=np.array(inputs), y=np.array(targets), y_labels = 40)
        hidden_layer = mlp.Tanh(layer_name='hidden', dim=40, irange=1)
        output_layer = mlp.Softmax(40, 'output', irange=1)
        layers = [hidden_layer, output_layer]
        ann = mlp.MLP(layers, nvis=207)
        trainer = Train(dataset = data,
                        model = ann,
                        algorithm = sgd.SGD(batch_size = 144,
                                            learning_rate = .01,
                                            monitoring_dataset = {
                                                'train' : data,
                                                'valid' : data,
                                                'test' : data
                                            },
                                            #cost = MyCostSubclass(),
                                            termination_criterion = ChannelTarget('valid_output_misclass',.001)),
                        save_path = "saved_models/Master_tetris_min_density.pkl",
                        save_freq = 1)
        trainer.main_loop()