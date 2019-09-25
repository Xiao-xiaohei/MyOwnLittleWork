#coding=utf-8

import os
import random
import pickle
import numpy as np
import torch as t

from utils.process import compute_vad_mask
from torch.nn.utils.rnn import pack_sequence, pad_sequence

# data is saved as ... '..data/speakers/[tr, ts, cv]/[mix1.npy, mix2.npy...]'

class MixSpeakers(object):
	def __init__(self, path, shuffle=True, vad_threshold=40):
		'''
		path is '.../total/[tr, cv, ts]'
		I'm too vegetable...
		'''
		self.root = path
		self.mixes = os.listdir(path)
		self.vad_threshold = vad_threshold
		try:
			self.mixes.remove('.DS_Store')
		except ValueError:
			pass
		self.length = len(self.mixes)
		if shuffle:
			random.shuffle(self.mixes)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		if isinstance(index, slice):
			tmps = []
			for ii in range(index.start, index.stop):
				tmps.append(self[ii])
			tmps = sorted(tmps, key=lambda x:x[0].shape[0], reverse=True)
			x_res = pack_sequence([x[0] for x in tmps])
			vad_res = pad_sequence([x[1] for x in tmps], batch_first=True)
			label_res = pad_sequence([x[2] for x in tmps], batch_first=True)
			return x_res, vad_res, label_res	#  which are not align
		else:
			mix = self.mixes[index]
			try:
				#     details of path       #
				x = np.load(os.path.join(self.root, mix))
				vad_x = compute_vad_mask(x[0], self.vad_threshold)

				#     details of label      #
				label = x[1:]

				#     trans to t.Tensor     #
				mix_x = t.from_numpy(x[0])
				vad_x = t.from_numpy(vad_x)
				label = t.from_numpy(label)
				label = label.permute(1, 0, 2)
				return [mix_x, vad_x, label]
			except:
				new_index = random.randint(0, len(self) - 1)
				return self[new_index]

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

			return (mixes, vads, labels)