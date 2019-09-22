#coding=utf-8

import torch as t
from config import opt
from utils.visualize import Visualizer
from utils.Trainer import Trainer
from itertools import permutations
import numpy as np
import os
import time

class RSHNetTrainer(Trainer):
	def __init__(self, opt):
		super(RSHNetTrainer, self).__init__(opt)
		self.alpha = opt.alpha
		self.beta = opt.beta
		self.greedy = opt.greedy

	def recursive_loss(self, data, label):
		'''
			data:
				mix [B, T, num_bins]
			label:
				M [C, B, T, num_bins]

			compute loss using PIT for M ~ M
			L_{Mask}: MSE?
			L_{flag}: CrossEntropy? MSE?
			L_{res_Mask}: Square？ or CrossEntropy that minus is heavily gg!
		'''
		C = label.shape[0]
		B = label.shape[1]
		stop_flag = t.zeros([B, C])
		stop_flag[:, -1] = 1
		loss = []
		flags = []
		M = t.ones(data.shape)	# M [B, T, num_bins]
		for i in range(C):
			inputs = t.cat([data, M], dim=-1)
			tmp_m, tmp_c = self.model(inputs)
			flags.append(tmp_c)

		L_flag = nn.BCELoss()(t.stack(flags, dim=1), stop_flag)
		L_resMask = t.norm(M, 2)
		

		'''
		output_flags = output[2]	# [B, C]
		L_flag = nn.BCELoss()(output_flags, label[1])

		L_resMask = t.norm(output[1], 2)
		M = output[0]
		C = M.shape[0]
		if self.greedy:
			L_mask = t.sum(output[3], dim=0)
		else:
			# pit_mat with shape [C!, B]
			pit_mat = t.stack([self.mse_loss(M, label[0], p) for p in permutations(range(C))])
			L_mask, min_per = t.min(pit_mat, dim=0)
		'''
		return L_mask + self.alpha * L_flag + self.beta * L_resMask

	def mse_loss(self, obtain_m, ref_m, permutation):
		'''
			obtain_m: the elimated masks [C, B, T, num_bins]
			ref_m: the reference masks [C, B, T, num_bins]
			permutation: one permutation of C!
		'''
		# get a loss with shape [B, ]
		return sum([self.mse(obtain_m[s], ref_m[t]) for s, t in enumerate(permutation)])

	def mse(self, ob_m, ref_m):
		'''
			input:
				ob_m: [B, T, num_bins]
				ref_m: [B, T, num_bins]
			out:
				loss: [B, ]
		'''
		return t.norm(ob_m - ref_m, p='fro', dim=[-2, -1])


def train(**kwargs):
	opt._parse(kwargs)
	
	trainer = RSHNetTrainer(opt)
	trainer.run()