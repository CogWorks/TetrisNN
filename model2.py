#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI"))

from simulator import *

import csv, json, gzip
import numpy as np
import random

import time

import theano
from pylearn2.models import mlp
from pylearn2.train import Train
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import ChannelTarget, EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial

# f = gzip.open('2014_Population_Study.tsv.gz','r')
# #f = open('test_data.tsv','r')
# dialect = csv.Sniffer().sniff(f.read(10240))
# f.seek(0)
# reader = csv.reader(f, dialect)
#
# header = reader.next()
#
pieces = ["I", "O", "T", "S", "Z", "J", "L"]
# data_array=[]
#
def to_one_hot(value, labels):
    return [1 if value==l else 0 for l in labels]
#
# def makeDDM(data_array):
#     X=np.array([d[0] for d in data_array])
#     y=np.array([d[1] for d in data_array])
#     return DenseDesignMatrix(X=X, y=y)
#
# for row in reader:
#     if row[1] == 'EP_SUMM':
#         board = json.loads(row[header.index('board_rep')])
#         col_heights = get_heights(board)
#         col_pits, pit_rows, lumped_pits = get_all_pits(board)
#         level = int(row[header.index('level')])
#         danger = int(json.loads(row[header.index('danger_mode')].lower()))
#         curr_zoid = to_one_hot(row[header.index('curr_zoid')], pieces)
#         features = col_heights + col_pits + [level, danger] + curr_zoid
#         zoid_rot = int(row[header.index('zoid_rot')])
#         zoid_col = int(row[header.index('zoid_col')])
#         action = [zoid_rot,zoid_col]
#         data_array.append((features, action))
#
# random.shuffle(data_array)
# data = makeDDM(data_array)
#
# hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=len(data_array[0][0]), irange=1000)
# output_layer = mlp.Softmax(2, 'output', irange=.1)
# layers = [hidden_layer, output_layer]
# ann = mlp.MLP(layers, nvis=len(data_array[0][0]))
#
# trainer = Train(dataset = data,
#                 model = ann,
#                 algorithm = sgd.SGD(batch_size = 1,
#                                     learning_rate = .001,
#                                     monitoring_dataset = {
#                                         'train' : data,
#                                         'valid' : data,
#                                         'test' : data
#                                     },
#                                     termination_criterion = ChannelTarget('valid_output_misclass',.2)))
# trainer.main_loop()

class NeuralNetController(TetrisController):
    """A tetris controller that picks an action using a neural net."""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, sim):
    	board = [0 if sim.space[r][c]==0 else 1 for r in range(0,20) for c in range(0,10)]
        zoid = to_one_hot(sim.curr_z, pieces)
    	
        cols = get_cols(sim.space)
        heights = get_heights(cols)
        ph = max(heights)
        profile = [ph-h for h in heights]
        col_sums = [sum([0 if cc==0 else 1 for cc in c]) for c in cols]
        penalty = [heights[i]-col_sums[i] for i in range(0,len(cols))]
        w = [i for i in range(0,len(heights)) if heights[i]>0]
        density = 1
        if len(w) > 0:
            ii = min(w)
            jj = max(w)
            width = jj - ii + 1
            height = max(heights)
            s = sum([sum([1 if r>0 else 0 for r in row]) for row in sim.space])
            density = 1.0*(s-sum(penalty))/(width*height)
        features = board + zoid
    	
    	inputs = np.array([features])
    	actions = self.model.fprop(theano.shared(inputs, name='inputs')).eval()[0]
    	
    	ranks = np.argsort(actions)[::-1]
    	for i in range(0,len(ranks)):
    		action = ranks[i]
    		rot = int(action) / 10
    		col = action - (rot * 10)
    		for option in sim.options:
    			if col == option[0] and rot == option[1]:
    				return option
        print ("SHIT")
        time.sleep(2)
        return random.choice(sim.options)

    def _print(self, feats=False):
        pass
        
while True:
    ann = serial.load("saved_models/Master_tetris_min_density.pkl")
    controller = NeuralNetController(ann)
    sim = TetrisSimulator(controller = controller,
    			show_choice = True, 
    			show_options = True, 
    			option_step = .3, 
    			choice_step = 1,
                seed = random.randint(1,100000000))

    sim.show_options = False
    sim.choice_step = 0
    sim.overhangs = False
    sim.force_legal = True
    sim.run()
    time.sleep(1)