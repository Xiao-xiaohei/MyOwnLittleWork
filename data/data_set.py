#coding=utf-8

import torch as t
import os
import random
import pickle
import numpy as np
from utils.process import compute_vad_mask

# data is saved as ... '../total/tr/A_B_xx/[s1/*.npy, s2/*.npy...]'

class MixSpeakers(object):
	def __init__(self, path, shuffle=True, vad_threshold=40):
		'''
		path is '.../total/[tr, cv, ts]'
		now just 2 speakers...orz, I'm too vegetable...
		'''
		self.mixes = os.listdir(path)
		self.vad_threshold = vad_threshold
		try:
			mixes.remove('.DS_Store')
		except ValueError:
			pass
		self.length = len(self.mixes)
		if shuffle:
			random.shuffle(self.mixes)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		if isinstance(index, slice):
			x_res = []
			vad_res = []
			label_res = []
			for ii in range(index.start, index.stop):
				tmp = self[ii]
				x_res.append(tmp[0])
				vad_res.append(tmp[1])
				label_res.append(tmp[2])
				return (x_res, vad_res, label_res)
		else:
			mix = self.mixes[index]
			try:
				#############################
				#     details of path       #
				#############################
				x = np.load(mix + '/mix.npy')
				vad_x = compute_vad_mask(x, self.vad_threshold)

			except:
				new_index = random.randint(0, len(self) - 1)

			#############################
			#     details of label      #
			#############################
			label = None

			#############################
			#     trans to t.Tensor     #
			#############################

			return (x, vad_x, label)

class DataLoader(object):
	def __init__(self, mix_speakers, batch_size=16, drop_last=False, vad_threshold=40):
		self.mix_speakers = mix_speakers
		self.batch_size = batch_size
		self.drop_last = drop_last
		self.vad_threshold = vad_threshold

	def __len__(self):
		if self.drop_last or (len(self.mix_speakers) % self.batch_size) == 0:
			return len(self.mix_speakers) // self.batch_size
		else:
			return len(self.mix_speakers) // self.batch_size + 1

	def __iter__(self):
		l = len(self)
		for step in range(0, l, self.batch_size):
			if step + self.batch_size > l:
				if self.drop_last:
					break
				else:
					b = l - self.batch_size
			else:
				b = step
			mixes, vads, labels = self.mix_speakers[step:step + self.batch_size]

			#############################
			#    pack or pad sequence   #
			#############################

			return (mixes, vads, labels)