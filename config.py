#coding=utf-8

import warnings
import torch as t

class DefaultConfig(object):
	# visdom
	vis = False 
	env = 'default'
	vis_port = 8097

	# model
	model = 'RSHNet'
	load_model_path = None
	checkpoint = './' + model + '_checkpoint'
	model_kwargs = {
		"num_bins":129,
		"hidden_size":600,
		"bidirectional":True,
		"num_layers":2,
		"rnn":"lstm",
		"act_func":"sigmoid"
	}
	# train args
	alpha = 1.
	beta = 1.
	greedy = True

	# dataset path ...
	train_identification_data_root = '../processed_data/train_identification'  # 训练集存放路径
	test_identification_data_root = '../processed_data/test_identification'  # 测试集存放路径

	batch_size = 32  # batch size
	use_gpu = False	# user GPU or not
	num_workers = 4  # how many workers for loading data
	print_freq = 20  # print info every N batch

	max_epoch = 15
	lr = 0.001  # learning rate
	weight_decay = 1e-5

	def _parse(self, kwargs):
		"""
		根据字典kwargs 更新 config参数
		"""
		for k, v in kwargs.items():
			if not hasattr(self, k):
				warnings.warn("Warning: opt has not attribut %s" % k)
			setattr(self, k, v)
		
		opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')


		print('user config:')
		for k, v in self.__class__.__dict__.items():
			if not k.startswith('_'):
				print(k, getattr(self, k))

opt = DefaultConfig()