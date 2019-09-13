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
		self.alpha = opt.alpha
		self.beta = opt.beta

	def loss(self, output, label):
		'''
			output:
				M [C, B, T, num_bins]
				res_M [B, T, num_bins]
				and
				z [B, C]
			label:
				M [B, C, T, num_bins]
				z [B, C]

			compute loss using PIT for M ~ M
			L_{Mask}: MSE?
			L_{flag}: CrossEntropy? MSE?
			L_{res_Mask}: Square？ or CrossEntropy that minus is heavily gg!
		'''
		output_flags = output[1]	# [B, C]
		L_flag = nn.BCELoss()(output[1], label[1])
		L_resMask = t.norm(output[0][-1], 2)


def train(**kwargs):
	opt._parse(kwargs)
	
	trainer = RSHNetTrainer(opt)
	trainer.run()