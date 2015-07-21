#!/usr/bin/env python

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Tetris-AI"))

from simulator import *

import csv, json, gzip
import numpy as np
import random

import theano
import theano.tensor as T
from pylearn2.utils import safe_zip, safe_izip, wraps
from pylearn2.models import autoencoder
from pylearn2.train import Train
from pylearn2.utils.iteration import SubsetIterator
from pylearn2.training_algorithms import sgd
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.termination_criteria import ChannelTarget, EpochCounter
from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from theano.compat.python2x import OrderedDict
from pylearn2.space import CompositeSpace

def to_one_hot(value, labels):
    return [1 if value==l else 0 for l in labels]

def makeDDM(data_array):
    X=np.array([d[0] for d in data_array])
    y=np.array([d[1] for d in data_array])
    return DenseDesignMatrix(X=X, y=y, y_labels=40)
    
def process_row(row, header):
    board = [1 if item>0 else 0 for sublist in json.loads(row[header.index('board_rep')]) for item in sublist]
    curr_zoid = to_one_hot(row[header.index('curr_zoid')], pieces)
    next_zoid = to_one_hot(row[header.index('next_zoid')], pieces)
    features = board + curr_zoid + next_zoid
    zoid_rot = int(row[header.index('zoid_rot')])
    zoid_col = int(row[header.index('zoid_col')])
    action = [zoid_rot * 10 + zoid_col]

class NeuralNetController(TetrisController):
    """A tetris controller that picks an action using a neural net."""
    
    def reset(self, model):
        self.model = model
        self.log = []
    
    def __init__(self, model):
        self.reset(model)
        
    def evaluate(self, sim):
        board = [1 if item>0 else 0 for sublist in sim.space for item in sublist]   
        features = board + to_one_hot(sim.curr_z, pieces) + to_one_hot(sim.next_z, pieces)
        
        inputs = np.array([features])
        actions = self.model.reconstruct(theano.shared(inputs, name='inputs')).eval()[0]
        
        ranks = np.argsort(actions)[::-1]
        
        choice = None
        for i in range(0,len(ranks)):
            action = ranks[i]
            rot = int(action) / 10
            col = action - (rot * 10)
            for option in sim.options:
                if col == option[0] and rot == option[1]:
                    choice = option
                    break
            if choice != None:
                break
        if choice == None:
            choice = random.choice(sim.options)
        
        self.log.append((features,[action]))
        
        return choice

    def _print(self, feats=False):
        pass

class TetrisSimulatorIterator(object):

    stochastic = False

    def __init__(self, dataset, seed=1):
        self._dataset = dataset
        self._num_batches = 1
        self._batch = 0
        self.controller = NeuralNetController(self._dataset.model)
        self.sim = TetrisSimulator(
            controller = self.controller,
            board = testboard(),
            curr=random.choice(pieces),
            next = random.choice(pieces),
            show_choice = False, 
            show_options = False, 
            option_step = .3, 
            choice_step = 1,
            seed = seed
        )
        self.sim.show_scores = False
        self.sim.show_result = False
        self.sim.show_options = False
        self.sim.choice_step = 0
        self.sim.overhangs = False
        self.sim.force_legal = True
        self._dataset.summary = self.sim.run(printstep=-1)
        print self.controller.log
        print len(self.controller.log)
        self.num_examples = len(self.controller.log)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
        
    def next(self):
        if self._batch < self._num_batches:
            self._batch += 1
            data = tuple(np.array([[d[0] for d in self.controller.log]]))
            print data
            return data
        else:
            raise StopIteration()

class TetrisSimulatorDataset(Dataset):
    
    def __init__(self, model):
        self.model = model
        self.summary = None
        
    @wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None,
                 return_tuple=False):
                 
        [mode, batch_size, num_batches, rng, data_specs] = self._init_iterator(
            mode, batch_size, num_batches, rng, data_specs)
            
        print (mode, batch_size, num_batches, rng, data_specs)
                 
        return TetrisSimulatorIterator(self)
        
    def get_num_examples(self, *args, **kwargs):
        return float('inf')

class TetrisCost(DefaultDataSpecsMixin, Cost):

    supervised = False
    
    def __init__(self, data):
        self.data = data

    def expr(self, model, data, **kwargs):
        if self.data.summary != None:
            return theano.shared(self.data.summary['score']*1.0)
        else:
            return theano.shared(0.0)

    def get_monitoring_channels(self, model, data, **kwargs):
        return OrderedDict([('score', self.expr(None,None))])

if __name__ == '__main__':

#   f = gzip.open('2014_Population_Study.tsv.gz','r')
#   f = open('test_data.tsv','r')
#   dialect = csv.Sniffer().sniff(f.read(10240))
#   f.seek(0)
#   reader = csv.reader(f, dialect)
#   header = reader.next()

#   data_array=[]
#   for row in reader:
#       if row[1] == 'EP_SUMM':
#           features, action = process_row(row, header)
#           data_array.append((features, action))
# 
#   all = makeDDM(data_array)

#   random.shuffle(data_array)
#   ntrain = int(.8*len(data_array))
#   nvalid = int(.1*len(data_array))
#   ntest = len(data_array)-ntrain-nvalid
#   train = makeDDM(data_array[0:ntrain])
#   valid = makeDDM(data_array[ntrain:(ntrain+nvalid)])
#   test = makeDDM(data_array[(ntrain+nvalid):])

    model = autoencoder.Autoencoder(nvis=214,nhid=2800,act_enc='tanh',act_dec='sigmoid',tied_weights=True)
    model.force_batch_size = 1
    data = TetrisSimulatorDataset(model)
    cost = TetrisCost(data)

    tc = EpochCounter(max_epochs = 10)
    train_algo = sgd.SGD(learning_rate = .05, cost = cost, termination_criterion = tc, train_iteration_mode="even_sequential")

    trainer = Train(dataset = data, model = model, algorithm = train_algo)
    trainer.main_loop()