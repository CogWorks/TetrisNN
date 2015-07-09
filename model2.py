#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI"))

from simulator import *

import csv, json, gzip
import numpy as np
import random

import theano
from pylearn2.models import mlp
from pylearn2.train import Train
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

f = gzip.open('2014_Population_Study.tsv.gz','r')
dialect = csv.Sniffer().sniff(f.read(10240))
f.seek(0)
reader = csv.reader(f, dialect)

header = reader.next()

pieces = ["I", "O", "T", "S", "Z", "J", "L"]
data_array=[]

def to_one_hot(value, labels):
	return [1 if value==l else 0 for l in labels]

def makeDDM(data_array):
    X=np.array([d[0] for d in data_array])
    y=np.array([d[1] for d in data_array])
    return DenseDesignMatrix(X=X, y=y, y_labels=40)

for row in reader:
	if row[1] == 'EP_SUMM':
		board = json.loads(row[header.index('board_rep')])
		col_heights = get_heights(board)
		col_pits, pit_rows, lumped_pits = get_all_pits(board)
		level = int(row[header.index('level')])
		curr_zoid = to_one_hot(row[header.index('curr_zoid')], pieces)
		next_zoid = to_one_hot(row[header.index('next_zoid')], pieces)
		features = col_heights + col_pits + [level] + curr_zoid + next_zoid
		zoid_rot = int(row[header.index('zoid_rot')])
		zoid_col = int(row[header.index('zoid_col')])
		action = [zoid_rot * 10 + zoid_col]
		data_array.append((features, action))

random.shuffle(data_array)

ntrain = int(.8*len(data_array))
nvalid = int(.1*len(data_array))
ntest = len(data_array)-ntrain-nvalid

train = makeDDM(data_array[0:ntrain])
valid = makeDDM(data_array[ntrain:(ntrain+nvalid)])
test = makeDDM(data_array[(ntrain+nvalid):])

hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=int((40+len(data_array[0][0]))/2), irange=.1, init_bias=1.0)
output_layer = mlp.Softmax(40, 'output', irange=.1)
layers = [hidden_layer, output_layer]
ann = mlp.MLP(layers, nvis=len(data_array[0][0]))

trainer = Train(dataset = train,
                model = ann,
                algorithm = sgd.SGD(batch_size = 500,
                                    learning_rate = .000001,
                                    monitoring_dataset = {
                                    	'train' : train,
                                    	'valid' : valid,
                                    	'test' : test
									},
                                    termination_criterion = EpochCounter(max_epochs=10)))
trainer.main_loop()

class NeuralNetController(TetrisController):
    """A tetris controller that picks an action using a neural net."""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, sim):
    	col_heights = get_heights(sim.space)
    	col_pits, pit_rows, lumped_pits = get_all_pits(sim.space)    	
    	features = col_heights + col_pits + [sim.level] + to_one_hot(sim.curr_z, pieces) + to_one_hot(sim.next_z, pieces)
    	
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
		return random.choice(sim.options)

    def _print(self, feats=False):
        pass
        
controller = NeuralNetController(ann)
    
sim = TetrisSimulator(controller = controller,
			board = testboard(), curr="L", next = "S",
			show_choice = True, 
			show_options = True, 
			option_step = .3, 
			choice_step = 1,
			seed = 1
			)

sim.show_options = False
sim.choice_step = 0
sim.overhangs = False
sim.force_legal = True 
sim.run()