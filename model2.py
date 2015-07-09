#!/usr/bin/env python

import numpy as np
import pprint

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


pp = pprint.PrettyPrinter(indent=4)

# These are the names and datatypes of all the episode logs
#probably no longer useful since I can't get them to work with numpy fromfile
#data_names=['ts',      'event_type',   'SID',      'ECID', 'session',  'game_type',    'game_number',  'episode_number',   'level',    'score',    'lines_cleared',    'completed',    'game_duration',    'avg_ep_duration',  'zoid_sequence',    'evt_id',   'evt_data1',    'evt_data2',    'curr_zoid',    'next_zoid',    'danger_mode',  'evt_sequence', 'rots', 'trans',    'path_length',  'min_rots', 'min_trans',                                                                                                    'min_path', 'min_rots_diff',    'min_trans_diff',   'min_path_diff',    'u_drops',  's_drops',  'prop_u_drops', 'initial_lat',  'drop_lat', 'avg_lat',  'tetrises_game',    'tetrises_level',   'delaying', 'dropping', 'zoid_rot', 'zoid_col', 'zoid_row',                                                                                                                 'board_rep',    'zoid_rep', 'smi_ts',   'smi_eyes', 'smi_samp_x_l', 'smi_samp_x_r', 'smi_samp_y_l', 'smi_samp_y_r', 'smi_diam_x_l', 'smi_diam_x_r', 'smi_diam_y_l', 'smi_diam_y_r', 'smi_eye_x_l',  'smi_eye_x_r',  'smi_eye_y_l',  'smi_eye_y_r',  'smi_eye_z_l',  'smi_eye_z_r',  'fix_x',    'fix_y',    'all_diffs',    'all_ht',   'all_trans',    'cd_1', 'cd_2', 'cd_3', 'cd_4', 'cd_5', 'cd_6', 'cd_7', 'cd_8', 'cd_9', 'cleared',  'col_trans',    'column_9', 'cuml_cleared', 'cuml_eroded',  'cuml_wells',   'd_all_ht', 'd_max_ht', 'd_mean_ht',                                                                                                                                                            'd_pits',   'deep_wells',   'eroded_cells', 'full_cells',   'jaggedness',   'landing_height',   'lumped_pits',  'matches',  'max_diffs',    'max_ht',   'max_ht_diff',  'max_well', 'mean_ht',  'mean_pit_depth',   'min_ht',   'min_ht_diff',  'move_score',   'nine_filled',  'pattern_div',  'pit_depth',    'pit_rows', 'pits', 'row_trans',    'tetris',   'tetris_progress',  'weighted_cells',   'wells']
#data_types=[float,     str,            int,        str,    str,        str,            int,            int,                int,        int,        int,                str,            str,                str,                str,                str,        str,            str,            str,            str,            bool,           str,            int,    int,        int,            int,        int,                                                                                                            int,        int,                int,                int,                int,        int,        float,          int,            int,        float,      int,                int,                str,        str,        int,        int,        int,                                                                                                                        str,            str,        str,        str,        str,            str,            str,            str,            str,            str,            str,            str,            str,            str,            str,            str,            str,            str,            str,        str,        int,            int,        int,            int,    int,    int,    int,    int,    int,    int,    int,    int,    int,        int,            int,        int,            int,            int,            int,        int,        float,                                                                                                                                                                  int,        int,            int,            int,            int,            int,                int,            int,        int,            int,        float,          int,        float,      float,              int,        float,          int,            int,            int,            int,            int,        int,    int,            int,        int,                int,                int]
#data_types_full=np.dtype([('ts',float),('event_type',str),('SID',int),('ECID',str),('session',str),('game_type',str),('game_number',int),('episode_number',int),('level',int),('score',int),('lines_cleared',int),('completed',str),('game_duration',str),('avg_ep_duration',str),('zoid_sequence',str),('evt_id',str),('evt_data1',str),('evt_data2',str),('curr_zoid',str),('next_zoid',str),('danger_mode',bool),('evt_sequence',str),('rots',int),('trans',int),('path_length',int),('min_rots',int),('min_trans',int),('min_path',int),('min_rots_diff',int),('min_trans_diff',int),('min_path_diff',int),('u_drops',int),('s_drops',int),('prop_u_drops',float),('initial_lat',int),('drop_lat',int),('avg_lat',float),('tetrises_game',int),('tetrises_level',int),('delaying',str),('dropping',str),('zoid_rot',int),('zoid_col',int),('zoid_row',int),('board_rep',str),('zoid_rep',str),('smi_ts',str),('smi_eyes',str),('smi_samp_x_l',str),('smi_samp_x_r',str),('smi_samp_y_l',str),('smi_samp_y_r',str),('smi_diam_x_l',str),('smi_diam_x_r',str),('smi_diam_y_l',str),('smi_diam_y_r',str),('smi_eye_x_l',str),('smi_eye_x_r',str),('smi_eye_y_l',str),('smi_eye_y_r',str),('smi_eye_z_l',str),('smi_eye_z_r',str),('fix_x',str),('fix_y',str),('all_diffs',int),('all_ht',int),('all_trans',int),('cd_1',int),('cd_2',int),('cd_3',int),('cd_4',int),('cd_5',int),('cd_6',int),('cd_7',int),('cd_8',int),('cd_9',int),('cleared',int),('col_trans',int),('column_9',int),('cuml_cleared',int),('cuml_eroded',int),('cuml_wells',int),('d_all_ht',int),('d_max_ht',int),('d_mean_ht',float),('d_pits',int),('deep_wells',int),('eroded_cells',int),('full_cells',int),('jaggedness',int),('landing_height',int),('lumped_pits',int),('matches',int),('max_diffs',int),('max_ht',int),('max_ht_diff',float),('max_well',int),('mean_ht',float),('mean_pit_depth',float),('min_ht',int),('min_ht_diff',float),('move_score',int),('nine_filled',int),('pattern_div',int),('pit_depth',int),('pit_row',int),('row_trans',int),('tetris',int),('tetris_progress',int),('weighted_cells',int),('wells',int)])

f = open('test_data.tsv','r')

f.readline()
#skip the first line because it just lists the log variables

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
    
def process_datalist(datalist, NFEATURES):
	X = [-1]*NFEATURES
	j=0
	for item in datalist[0]:
		X[j]=item
		j+=1
	X[j]=datalist[1]
	j+=1
	for item in zoid_to_binary(pieces,datalist[2]):
		X[j]=item
		j+=1
	y = [datalist[3][0] * 10 + datalist[3][1]]
	return (X, y)

def GetInputOutput(data_array,pieces):
    #Changes python array of the relevant data to a DenseDesignMatrix, required for mlp.mlp
    lda = len(data_array)
    X=[None]*lda
    y=[None]*lda
    for i in range(0,lda):
        X[i], y[i] = process_datalist(data_array[i],NFEATURES)
    X=np.array(X)
    y=np.array(y)
    return DenseDesignMatrix(X=X, y=y, y_labels=40)


#67 through 75 is height change
#99 is min_ht
#18 is curr_zoid
#41,42 is zoid_rot,col

lines = f.readlines()
random.shuffle(lines)

for line in lines:
    temp_array=[]
    sub_temp_array=[]
    line_array = line.split(chr(9))
    if (line_array[1] == "EP_SUMM"):    
        for x in range(67,75):
            sub_temp_array.append(int(line_array[x]))       
        temp_array.append(sub_temp_array)       #column jaggedness
        temp_array.append(int(line_array[99]))  #min_ht
        temp_array.append(line_array[18])       #zoid
        temp_array.append((int(line_array[41]),int(line_array[42])))        #position, orientation      
        data_array.append(temp_array)

ntrain = int(.9*len(data_array))
nvalid = int(.09*len(data_array))
ntest = len(data_array)-ntrain-nvalid

train = GetInputOutput(data_array[0:ntrain], pieces)
valid = GetInputOutput(data_array[ntrain:(ntrain+nvalid)], pieces)
test = GetInputOutput(data_array[(ntrain+nvalid):], pieces)

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

model = LogisticRegression(nvis=17,nclasses=40)

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

for d in data_array:
	X, y = process_datalist(d,NFEATURES)
	print ("input", X, y)
	print ("output", model.logistic_regression(X).argmax(axis=1).eval())