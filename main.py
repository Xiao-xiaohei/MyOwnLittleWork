#coding=utf-8

import torch as t
from config import opt
from utils.visualize import Visualizer
from utils.Trainer import Trainer
import numpy as np
import os
import time

class RSHNetTrainer(Trainer):
	def __init__(self, opt):
		super(RSHNetTrainer, self).__init__(opt)

	def loss(self, output, input, alpha=1., beta=1.):
		'''
			output:
				M [C + 1, B, T, num_bins]
				and
				z [z_1, ..., z_C]
			input or label:
				M [C, B, T, num_bins]
				z [z_1, ..., z_C]

			compute loss using PIT for M ~ M
			L_{Mask}: MSE?
			L_{flag}: CrossEntropy? MSE?
			L_{res_Mask}: Square？ 
		'''

def train(**kwargs):
	opt._parse(kwargs)
	
	trainer = RSHNetTrainer(opt)
	trainer.run()