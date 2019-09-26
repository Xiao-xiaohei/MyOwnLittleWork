#coding=utf-8

import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
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
				mix [B, T, num_bins]	PackedSequence...
			label:
				M [C, B, T, num_bins]	Padded...

			compute loss using PIT for M ~ M
			L_{Mask}: MSE?
			L_{flag}: CrossEntropy? MSE?
			L_{res_Mask}: Square？ or CrossEntropy that minus is heavily gg!
		'''
		C = label.shape[0]
		B = label.shape[1]
		num_bins = label.shape[-1]
		stop_flag = t.zeros([B, C])
		stop_flag[:, -1] = 1
		loss = []	# if greedy, it's directly loss array [C, B], else Ms [C, B, T, num_bins]!
		flags = []
		padded_data, data_lengths = pad_packed_sequence(data, batch_first=True)
		Loss_Mask = pad_sequence([t.ones([times, C, num_bins]) for times in data_lengths], batch_first=True)	# [B, T, C, num_bins]
		Loss_Mask = Loss_Mask.permute(2, 0, 1, 3)	# [C, B, T, num_bins]
		M = t.ones(padded_data.shape)	# M [B, T, num_bins]
		res = t.ones([C, B])
		min_per = []
		for i in range(C):
			inputs = pack_padded_sequence(t.cat([padded_data, M], dim=-1), data_lengths, batch_first=True)
			tmp_m, tmp_z = self.model(inputs)
			if self.greedy:
				tmp_M = t.stack([tmp_m for _ in range(C)], dim=0)
				tmp_loss = t.norm((tmp_M - label) * Loss_Mask, p='fro', dim=[-2, -1])	# size [C, B]
				# weight mask the tmp_loss (since some have been matched
				tmp_loss += (t.max(tmp_loss) * t.ones(tmp_loss.shape))
				# get indices
				indice = t.min(tmp_loss, dim=0)	# both values [B, ] and indices [B, ]
				min_per.append(indice.indices)
				new_mask = []
				for iii in range(B):
					new_mask.append(label[indice.indices[iii]][iii])
					res[indice.indices[iii]][iii] = 0

				M -= t.stack(new_mask, dim=0)
				loss.append(indice.values)
			else:
				M -= tmp_m
				loss.append(tmp_m)
			flags.append(tmp_z)

		loss = t.stack(loss, dim=0)	# losses [C, B] or Ms [C, B, T, num_bins]	

		if self.greedy:
			min_per = t.stack(min_per, dim=0)
			L_mask = t.sum(loss)
		else:
			pit_mat = t.stack([self.mse_loss(loss, label, p, Loss_Mask) for p in permutations(range(C))])
			L_mask, min_per = t.min(pit_mat, dim=0)
			L_mask = t.sum(L_mask)

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
		# for test
		return L_mask, L_flag, L_resMask

		#return L_mask + self.alpha * L_flag + self.beta * L_resMask

	def mse_loss(self, obtain_m, ref_m, permutation, Mask):
		'''
			obtain_m: the elimated masks [C, B, T, num_bins]
			ref_m: the reference masks [C, B, T, num_bins]
			permutation: one permutation of C!
			Mask: deal with differrent T steps... [C, B, T, num_bins]
		'''
		# get a loss with shape [B, ]
		return sum([self.mse(obtain_m[s], ref_m[t], Mask[t]) for s, t in enumerate(permutation)])

	def mse(self, ob_m, ref_m, loss_m):
		'''
			input:
				ob_m: [B, T, num_bins]
				ref_m: [B, T, num_bins]
				loss_m: [B, T, num_bins]
			out:
				loss: [B, ]
		'''
		return t.norm((ob_m - ref_m) * loss_m, p='fro', dim=[-2, -1])


def train(**kwargs):
	opt._parse(kwargs)
	
	trainer = RSHNetTrainer(opt)
	trainer.run()