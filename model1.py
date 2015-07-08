
import numpy as np
import pprint

import itertools

import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

pp = pprint.PrettyPrinter(indent=4)


#data_names=['ts',		'event_type',	'SID',		'ECID',	'session',	'game_type',	'game_number',	'episode_number',	'level',	'score',	'lines_cleared',	'completed',	'game_duration',	'avg_ep_duration',	'zoid_sequence',	'evt_id',	'evt_data1',	'evt_data2',	'curr_zoid',	'next_zoid',	'danger_mode',	'evt_sequence',	'rots',	'trans',	'path_length',	'min_rots',	'min_trans',																									'min_path',	'min_rots_diff',	'min_trans_diff',	'min_path_diff',	'u_drops',	's_drops',	'prop_u_drops',	'initial_lat',	'drop_lat',	'avg_lat',	'tetrises_game',	'tetrises_level',	'delaying',	'dropping',	'zoid_rot',	'zoid_col',	'zoid_row',																													'board_rep',	'zoid_rep',	'smi_ts',	'smi_eyes',	'smi_samp_x_l',	'smi_samp_x_r',	'smi_samp_y_l',	'smi_samp_y_r',	'smi_diam_x_l',	'smi_diam_x_r',	'smi_diam_y_l',	'smi_diam_y_r',	'smi_eye_x_l',	'smi_eye_x_r',	'smi_eye_y_l',	'smi_eye_y_r',	'smi_eye_z_l',	'smi_eye_z_r',	'fix_x',	'fix_y',	'all_diffs',	'all_ht',	'all_trans',	'cd_1',	'cd_2',	'cd_3',	'cd_4',	'cd_5',	'cd_6',	'cd_7',	'cd_8',	'cd_9',	'cleared',	'col_trans',	'column_9',	'cuml_cleared',	'cuml_eroded',	'cuml_wells',	'd_all_ht',	'd_max_ht',	'd_mean_ht',																																							'd_pits',	'deep_wells',	'eroded_cells',	'full_cells',	'jaggedness',	'landing_height',	'lumped_pits',	'matches',	'max_diffs',	'max_ht',	'max_ht_diff',	'max_well',	'mean_ht',	'mean_pit_depth',	'min_ht',	'min_ht_diff',	'move_score',	'nine_filled',	'pattern_div',	'pit_depth',	'pit_rows',	'pits',	'row_trans',	'tetris',	'tetris_progress',	'weighted_cells',	'wells']
#data_types=[float,		str,			int,		str,	str,		str,			int,			int,				int,		int,		int,				str,			str,				str,				str,				str,		str,			str,			str,			str,			bool,			str,			int,	int,		int,			int,		int,																											int,		int,				int,				int,				int,		int,		float,			int,			int,		float,		int,				int,				str,		str,		int,		int,		int,																														str,			str,		str,		str,		str,			str,			str,			str,			str,			str,			str,			str,			str,			str,			str,			str,			str,			str,			str,		str,		int,			int,		int,			int,	int,	int,	int,	int,	int,	int,	int,	int,	int,		int,			int,		int,			int,			int,			int,		int,		float,																																									int,		int,			int,			int,			int,			int,				int,			int,		int,			int,		float,			int,		float,		float,				int,		float,			int,			int,			int,			int,			int,		int,	int,			int,		int,				int,				int]

#data_types_full=np.dtype([('ts',float),('event_type',str),('SID',int),('ECID',str),('session',str),('game_type',str),('game_number',int),('episode_number',int),('level',int),('score',int),('lines_cleared',int),('completed',str),('game_duration',str),('avg_ep_duration',str),('zoid_sequence',str),('evt_id',str),('evt_data1',str),('evt_data2',str),('curr_zoid',str),('next_zoid',str),('danger_mode',bool),('evt_sequence',str),('rots',int),('trans',int),('path_length',int),('min_rots',int),('min_trans',int),('min_path',int),('min_rots_diff',int),('min_trans_diff',int),('min_path_diff',int),('u_drops',int),('s_drops',int),('prop_u_drops',float),('initial_lat',int),('drop_lat',int),('avg_lat',float),('tetrises_game',int),('tetrises_level',int),('delaying',str),('dropping',str),('zoid_rot',int),('zoid_col',int),('zoid_row',int),('board_rep',str),('zoid_rep',str),('smi_ts',str),('smi_eyes',str),('smi_samp_x_l',str),('smi_samp_x_r',str),('smi_samp_y_l',str),('smi_samp_y_r',str),('smi_diam_x_l',str),('smi_diam_x_r',str),('smi_diam_y_l',str),('smi_diam_y_r',str),('smi_eye_x_l',str),('smi_eye_x_r',str),('smi_eye_y_l',str),('smi_eye_y_r',str),('smi_eye_z_l',str),('smi_eye_z_r',str),('fix_x',str),('fix_y',str),('all_diffs',int),('all_ht',int),('all_trans',int),('cd_1',int),('cd_2',int),('cd_3',int),('cd_4',int),('cd_5',int),('cd_6',int),('cd_7',int),('cd_8',int),('cd_9',int),('cleared',int),('col_trans',int),('column_9',int),('cuml_cleared',int),('cuml_eroded',int),('cuml_wells',int),('d_all_ht',int),('d_max_ht',int),('d_mean_ht',float),('d_pits',int),('deep_wells',int),('eroded_cells',int),('full_cells',int),('jaggedness',int),('landing_height',int),('lumped_pits',int),('matches',int),('max_diffs',int),('max_ht',int),('max_ht_diff',float),('max_well',int),('mean_ht',float),('mean_pit_depth',float),('min_ht',int),('min_ht_diff',float),('move_score',int),('nine_filled',int),('pattern_div',int),('pit_depth',int),('pit_row',int),('row_trans',int),('tetris',int),('tetris_progress',int),('weighted_cells',int),('wells',int)])

f = open('episodes_3001_2014-10-16_14-13-19.tsv','r')

line1 = f.readline()


pieces = ["I", "O", "T", "S", "Z", "J", "L"]
data_array=[]

def GetInputOutput(data_array,pieces):
	X=[[-1]*17]*len(data_array)
	y=[[-1]*2]*len(data_array)
	i=0
	for datalist in data_array:
		j=0
		for item in datalist[0]:
			X[i][j]=item
			j+=1
		X[i][j]=datalist[1]
		j+=1
		for item in zoid_to_binary(pieces,datalist[2]):
			X[i][j]=item
			j+=1
		k=0
		for item in datalist[3]:
			y[i][k]=item
			k+=1
		i+=1		
	print len(X[0])
	X=np.array(X)
	y=np.array(y)
	return DenseDesignMatrix(X=X, y=y)
'''
class GetInputOutputClass(DenseDesignMatrix):
    def __init__(self):
		X=[[-1]*17]*len(data_array)
		y=[[-1]*2]*len(data_array)
		i=0
		for datalist in data_array:
			for item in datalist[0]:
				X[i].append(item)
			X[i].append(datalist[1])
			for item in zoid_to_binary(pieces,datalist[2]):
				X[1].append(item)		
			for item in datalist[3]:
				y[i].append(item)
			i+=1		
		X=np.array(X)
		y=np.array(y)
		super(GetInputOutputClass, self).__init(X=X, y=y)

#^^^ Seems to be the general scheme for getting inputs into the requisite DenseDesignMatrix
#No idea why it throws a metaclass error when other examples using the same format do not.  
 '''

def zoid_to_binary(pieces,zoid):
	binary=[0]*len(pieces)
	i=0
	for z in pieces:
		if zoid==z:
			binary[i]=1
			break
		else:
			i+=1
	return binary


#67 through 75 is height change
#99 is min_ht
#18 is curr_zoid
#41,42 is zoid_rot,col

for line in f:
	temp_array=[]
	sub_temp_array=[]
	line_array = line.split(chr(9))
	if (line_array[1] == "EP_SUMM"):	
		for x in range(67,75):
			sub_temp_array.append(int(line_array[x]))		
		temp_array.append(sub_temp_array)		#column jaggedness
		temp_array.append(int(line_array[99]))	#min_ht
		temp_array.append(line_array[18])		#zoid
		temp_array.append((int(line_array[41]),int(line_array[42])))		#position, orientation		
		data_array.append(temp_array)

ds = GetInputOutput(data_array, pieces)

hidden_layer = mlp.Linear(layer_name='hidden', dim=10, irange=.1, init_bias=1.)

output_layer = mlp.Softmax(2, 'output', irange=.1)

trainer = sgd.SGD(learning_rate=.05, batch_size=10, termination_criterion=EpochCounter(1000))

layers = [hidden_layer, output_layer]
ann = mlp.MLP(layers, nvis=17)
trainer.setup(ann, ds)

weights = ann.get_weights()

while True:
    trainer.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()
    print "cost: ", ann.cost()
    if not trainer.continue_learning(ann):
        break
        
print "before"
print weights
print "after"      
print ann.get_weights()

m=0
for f,b in itertools.izip(weights,ann.get_weights()):
    print m, 
    print f
    print b
    m+=1


#Get cd_... and min_ht
#get zoid
#get position and orientation


def check_int(item):
	if isinstance(item, (int, long)):
		return True
	return False

def check_float(item):
	if isintance(item, float):
		return True
	return False
	


def lists_to_board(item_string):
	rows=item_string.strip("'[[").strip("]]'")
	rows_list=rows.split("], [")
	board_array=[]
	for row in rows_list:
		board_array.append(row.split(", "))
	return np.array(board_array)

def evt_sequence_to_list(evtsequence):
	evts=evtsequence.strip("[[").strip("]]")
	evts_list=evts.split("], [")
	evts_array=[]
	temp_list=()
	for event in evts_list:
		temp_list=event.split(", ")
		temp_list[0]=temp_list[0].strip("\"")
		evts_array.append(temp_list)
	return np.array(evts_array)

#teststr="'[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0, 2, 2]]'"
#teststr2="[[\"sy-dn-s\", 799], [\"kp-tr-l\", 1199], [\"sy-tr-l\", 1199], [\"kr-tr-l\", 1366], [\"sy-dn-s\", 1599], [\"kp-tr-l\", 2066], [\"sy-tr-l\", 2066], [\"kr-tr-l\", 2166], [\"sy-dn-s\", 2399], [\"kp-tr-l\", 2500], [\"sy-tr-l\", 2500], [\"kr-tr-l\", 2599], [\"kp-tr-l\", 2900], [\"sy-tr-l\", 2900], [\"kr-tr-l\", 3033], [\"sy-dn-s\", 3199], [\"kp-tr-r\", 3633], [\"sy-tr-r\", 3633], [\"sy-tr-r\", 3866], [\"sy-tr-r\", 3966], [\"sy-dn-s\", 3999], [\"sy-tr-r\", 4066], [\"sy-tr-r\", 4166], [\"sy-tr-r\", 4266], [\"sy-tr-r\", 4367], [\"sy-tr-r\", 4466], [\"sy-tr-r\", 4566], [\"kr-tr-r\", 4633], [\"sy-dn-s\", 4799], [\"sy-dn-s\", 5599], [\"sy-dn-s\", 6399], [\"kp-rt-cw\", 6433], [\"sy-rt-cw\", 6433], [\"kr-rt-cw\", 6599], [\"kp-rt-cc\", 7133], [\"sy-rt-cc\", 7133], [\"sy-dn-s\", 7199], [\"kr-rt-cc\", 7233], [\"kp-dwn\", 7966], [\"sy-dn-u\", 7966], [\"sy-dn-u\", 7999], [\"sy-dn-u\", 8033], [\"sy-dn-u\", 8066], [\"sy-dn-u\", 8099], [\"sy-dn-u\", 8133], [\"sy-dn-u\", 8166], [\"kr-dwn\", 8199], [\"sy-dn-s\", 8966], [\"kp-dwn\", 9199], [\"sy-dn-u\", 9199], [\"sy-dn-u\", 9233]]"


#21 is event sequence
#44 is board_rep
#45 is zoid rep

#FIRST LINE IS A LIST OF ALL THE LOG VARIABLES
#LAST LINE LISTS ALL THE ZOIDS WHICH DROPPED DURING THE GAME
#(IE GAME_SUMM)

