#coding=utf-8

import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

from config import opt
from utils.visualize import Visualizer
from utils.Trainer import Trainer
from utils.util import RebuildWavFromMask

from data import MixSpeakers_time as MixSpeakers
from data import DataLoader_time as DataLoader
from itertools import permutations
import numpy as np
import os
import time

class ConvTasNetTrainer(Trainer):
	def __init__(self, opt):
		super(RSHNetTrainer, self).__init__(opt)
		self.alpha = opt.alpha
		self.beta = opt.beta
		self.greedy = opt.greedy

	def set_train_dataloader(self):
		#mixes = MixSpeakers(self.opt.train_data_path.format(num=self.opt.speaker_nums[0], data_type='tr'))	# just 2 speakers now for test!
		#return DataLoader(mixes, batch_size=self.opt.batch_size), None
		return DataLoader(self.opt.train_data_path, self.opt.speaker_nums, self.opt.batch_size), DataLoader(self.opt.cv_data_path, self.opt.speaker_nums, self.opt.batch_size)

	def set_test_dataloader(self):
		return DataLoader(self.opt.test_data_path, self.opt.speaker_nums, batch_size=1)

	def recursive_loss(self, data, label):
		'''
			data:
				data [B, duration x samples=32000]
			label:
				S [B, C, N]

			compute loss using PIT for M ~ M
			now just PIT?
			OR-PIT remained...
		'''
		# just for test... no need of vad...
		C = label.shape[1]
		B = label.shape[0]
		label = label.permute(1, 0, 2)	# [C, B, N]
		loss = []	# if greedy/OR-PIT, it's directly loss array [C, B], else S_ [C, B, N]!
		min_per = []
		for i in range(C):
			rebuild_s = self.model(data)[0]
			if self.greedy:
				pass
			else:
				data = data - rebuild_s
				loss.append(rebuild_s)

		loss = t.stack(loss, dim=0)	# losses [C, B] or S_ [C, B, T, N]	

		if self.greedy:
			pass
		else:
			pit_mat = t.stack([self.si_sdr_loss(loss, label, p) for p in permutations(range(C))])
			L_SiSNR, min_per = t.min(pit_mat, dim=0)
			L_SiSNR = t.sum(L_SiSNR)
		
		return L_SiSNR

	def sisnr_loss(self, obtain_s, ref_s, permutation):
		'''
			obtain_m: the elimated masks [C, B, N]
			ref_m: the reference masks [C, B, N]
			permutation: one permutation of C!
		'''
		# get a loss with shape [B, ]
		return sum([self.sisnr(obtain_m[s], ref_m[t]) for s, t in enumerate(permutation)])

	def sisnr(self, ob_s, ref_s, normalize=True):
		'''
			ob_s: [B, N]
			ref_s: [B, N]
		'''
		def vec_l2norm(x):
			return np.linalg.norm(x, 2, axis=1)	# [B, ]
		if normalize:
			n_ob_s = ob_s - ob_s.mean(axis=1)
			n_ref_s = ref_s - ref_s.mean(axis=1)
			tar = (n_ob_s * n_ref_s).sum(axis=1) * n_ref_s / vec_l2norm(n_ref_s) ** 2
			noi = n_ob_s - tar
		else:
			tar = (ob_s * ref_s).sum(axis=1) * ref_s / vec_l2norm(ref_s) ** 2
			noi = ob_s - tar
		return 20 * np.log10(vec_l2norm(tar)/vec_l2norm(noi))


	def compute_evaluation(self, datas, label, types):
		'''
			datas:
				mix: [N, ]
			label:
				[np arrays [C, N], sdrs [sdr_1, sdr_2, ...]]
			types:
				['Acc', 'SDR', ...]

		'''
		datas = datas.to(self.opt.device, dtype=t.float32)
		ori_sdr = label[1]
		C = len(label[0])
		c = 0
		label = np.stack(label[0], axis=0)

		compute_SDR = True
		if 'SDR' not in types:
			compute_SDR = False
		compute_SIR = True
		if 'SDR' not in types:
			compute_SIR = False
		compute_SAR = True
		if 'SDR' not in types:
			compute_SAR = False

		SIRs = []
		SDRs = []
		SARs = []
		c_SDRs = {}
		Accs = []
		while c < C:
			rebuild_s = self.model(datas)
			tmp_rebuild_s = np.stack([rebuild_s for _ in range(C)], axis=0)	# [C, N]

			if compute_SDR or compute_SIR compute_SAR:
				sdrs, sirs, sars, _ = bss_eval_sources(label, tmp_rebuild_s, compute_permutation=False)
				sdrs -= ori_sdr

			if compute_SDR:
				tmp_ans = max(sdrs)
				SDRs.append(tmp_ans)
				c_SDRs[c] = tmp_ans

			if compute_SIR:
				SIRs.append(max(sirs))
			if compute_SAR:
				SARs.append(max(sars))

			c += 1
			datas -= rebuild_s

		assert c == C

		if 'Acc' in types:
			if c == C:
				ans['Acc'] = True
			else:
				ans['Acc'] = False
			ans['Accs'] = Accs
		if compute_SDR:
			ans['SDR'] = np.sum(SDRs) / C
			ans['c_SDR'] = c_SDRs
		if compute_SIR:
			ans['SIR'] = np.mean(SIRs)
		if compute_SAR:
			ans['SAR'] = np.mean(SARs)

		return ans

def train(**kwargs):
	opt._parse(kwargs)
	
	trainer = RSHNetTrainer(opt)
	trainer.run()

def test(**kwargs):
	opt._parse(kwargs)
	
	trainer = RSHNetTrainer(opt)
	trainer.test()

def help():
	print("""
	usage : python file.py <function> [--args=value]
	<function> := train | help
	example: 
			python {0} train --env='env0701' --lr=0.01
			python {0} help
	avaiable args:""".format(__file__))

	from inspect import getsource
	source = (getsource(opt.__class__))
	print(source)

if __name__=='__main__':
	import fire
	fire.Fire()