#coding=utf-8

import os
import random
import pickle
import numpy as np
import torch as t

from utils.process import read_wav
from torch.nn.utils.rnn import pack_sequence, pad_sequence

# data is saved as ... '.../data/Cspeakers/.../[tr, ts, cv]/[mix, s1, s2, ...]/*.wav'

class MixSpeakers(object):
	def __init__(self, path, samplerate=8000, duration=4, train=True, shuffle=True):
		'''
		samplerate * duration(s) = datalength
		path: .../Cspeakers/.../[tr, ...]/
		'''
		tmp_dir = os.listdir(path)
		self.train = train
		self.C = len(tmp_dir) - 2
		self.mixes = os.listdir(path + '/mix')
		try:
			self.mixes.remove('.DS_Store')
		except ValueError:
			pass
		self.length = len(self.mixes)
		self.datalength = samplerate * duration
		self.path = path
		if shuffle:
			random.shuffle(self.mixes)

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		if isinstance(index, slice):
			datas = []
			for ii in range(index.start, index.stop):
				datas.append(self[ii])
			return t.stack(datas, dim=0)	# [B, C + 1, N=datalength]
		else:
			wav_name = self.mixes[index]
			mix_path = os.path.join(self.path + '/mix', wav_name)
			label_paths = [os.path.join(self.path + '/{}s'.format(i + 1), wav_name) for i in range(self.C)]
			mix = read_wav(mix_path)	# [1, N]
			if len(mix) < self.datalength:
				return self[random.randint(0, len(self) - 1)]
			labels = [read_wav(label_path) for label_path in label_paths]
			data = np.stack([mix] + labels, axis=0)
			return t.from_numpy(data[:, :self.datalength])	# [C + 1, N=datalength]

def get_bs(l, batch_size, index, drop_last=False):
	ans = [(s, index) for s in range(0, l, batch_size)]
	if ans[-1][0] + batch_size > l:
		if drop_last:
			ans = ans[:-1]
		else:
			ans[-1] = (l - batch_size, index)
	return ans

class DataLoader(object):
	def __init__(self, path, speaker_nums, samplerate=8000, duration=4, batch_size=16, drop_last=False, min_scale='batch'):
		'''
			path: .../{num}speakers/data_type
			min_scale: 'subset' means [2s, 3s, 4s, ...] or 'batch' means [batch_2s, batch_3s, batch_2s, batch_4s, ...]
		'''
		self.batch_size = batch_size
		if min_scale not in ['subset', 'batch']:
			raise ValueError("No such scale for dataloading!")
		self.train = True
		if path.split('/')[-1] == 'ts':
			self.train = False
			assert batch_size == 1

		self.mixes = [MixSpeakers(path.format(num=c), samplerate, duration, self.train) for c in speaker_nums]
		self.indices = []
		for ii, mix in enumerate(self.mixes):
			self.indices += get_bs(len(mix), batch_size, ii, drop_last)

		if min_scale == 'batch':
			random.shuffle(self.indices)

	def __len__(self):
		return len(self.indices)

	def __iter__(self):
		for b, index in self.indices:
			if self.train:
				data = self.mixes[index][b:b + self.batch_size]
			else:
				data = self.mixes[index][b]

			yield data