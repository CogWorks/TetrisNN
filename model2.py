#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI"))
from simulator import *

import csv, json

import numpy as np

import random

import itertools

import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

import numpy
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.train import Train
from theano.compat.python2x import OrderedDict
from pylearn2.space import CompositeSpace

f = open('test_data.tsv','r')
dialect = csv.Sniffer().sniff(f.read(10240))
f.seek(0)
reader = csv.reader(f, dialect)

header = reader.next()
print(header)

pieces = ["I", "O", "T", "S", "Z", "J", "L"]
data_array=[]

NFEATURES = 17

def zoid_to_binary(pieces,zoid):
    #Turns the zoid into a binary classification representation based on the pieces list
    binary=[0]*len(pieces)
    i=0
    for z in pieces:
        if zoid==z:
            binary[i]=1
            break
        else:
            i+=1
    return binary

def makeDDM(data_array):
    X=np.array([d[0] for d in data_array])
    y=np.array([d[1] for d in data_array])
    return DenseDesignMatrix(X=X, y=y, y_labels=40)

for row in reader:
	board = row[header.index('board_rep')]
	if board != '':
		board = json.loads(board)
		col_heights = get_heights(board)
		col_pits, pit_rows, lumped_pits = get_all_pits(board)
		level = int(row[header.index('level')])
		curr_zoid = zoid_to_binary(pieces, row[header.index('curr_zoid')])
		next_zoid = zoid_to_binary(pieces, row[header.index('next_zoid')])
		features = col_heights + col_pits + [level] + curr_zoid + next_zoid
		zoid_rot = int(row[header.index('zoid_rot')])
		zoid_col = int(row[header.index('zoid_col')])
		action = [zoid_rot * 10 + zoid_col]
		data_array.append((features, action))
		print(data_array[-1])

random.shuffle(data_array)

ntrain = int(.9*len(data_array))
nvalid = int(.09*len(data_array))
ntest = len(data_array)-ntrain-nvalid

train = makeDDM(data_array[0:ntrain])
valid = makeDDM(data_array[ntrain:(ntrain+nvalid)])
test = makeDDM(data_array[(ntrain+nvalid):])

class LogisticRegression(Model):
    def __init__(self, nvis, nclasses):
        super(LogisticRegression, self).__init__()

        self.nvis = nvis
        self.nclasses = nclasses

        W_value = numpy.random.uniform(size=(self.nvis, self.nclasses))
        self.W = sharedX(W_value, 'W')
        b_value = numpy.zeros(self.nclasses)
        self.b = sharedX(b_value, 'b')
        self._params = [self.W, self.b]

        self.input_space = VectorSpace(dim=self.nvis)
        self.output_space = VectorSpace(dim=self.nclasses)

    def logistic_regression(self, inputs):
        return T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        
    def get_monitoring_data_specs(self):
        space = CompositeSpace([self.get_input_space(),
                                self.get_target_space()])
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_monitoring_channels(self, data):
        space, source = self.get_monitoring_data_specs()
        space.validate(data)

        X, y = data
        y_hat = self.logistic_regression(X)
        error = T.neq(y.argmax(axis=1), y_hat.argmax(axis=1)).mean()
        return OrderedDict([('error', error)])
        
class LogisticRegressionCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)
        
        inputs, targets = data
        outputs = model.logistic_regression(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()

model = LogisticRegression(nvis=len(data_array[0][0]),nclasses=40)

trainer = Train(dataset = train,
                model = model,
                algorithm = sgd.SGD(batch_size = 200,
                                    learning_rate = 1e-3,
                                    monitoring_dataset = {
                                    	'train' : train,
                                    	'valid' : valid,
                                    	'test' : test
									},
                                    cost = LogisticRegressionCost(),
                                    termination_criterion = EpochCounter(max_epochs=100)))
trainer.main_loop()

class NeuralNetController(TetrisController):
    """A tetris controller that picks an action using a neural net."""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate(self, sim):
    	options = sim.options
    	col_heights = get_heights(sim.space)
    	col_pits, pit_rows, lumped_pits = get_all_pits(sim.space)    	
    	features = col_heights + col_pits + [sim.level] + zoid_to_binary(pieces, sim.curr_z) + zoid_to_binary(pieces, sim.next_z)
    	action = model.logistic_regression(features).argmax(axis=1).eval()[0]
    	rot = int(action) / 10
    	col = action - (rot * 10)
    	row = sim.find_drop(col, rot, sim.curr_z, sim.space)
    	simboard, ends_game = sim.possible_board(col, rot, row, zoid=sim.curr_z)
    	features = sim.get_features(simboard, prev_space=sim.space, all = False)
    	opt = [col,rot,row,simboard,features,ends_game]
    	print(opt)
    	return opt

    def _print(self, feats=False):
        pass
        
controller = NeuralNetController(model)
    
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

# for d in data_array:
# 	X, y = process_datalist(d,NFEATURES)
# 	print ("input", X, y)
# 	print ("output", model.logistic_regression(X).argmax(axis=1).eval())