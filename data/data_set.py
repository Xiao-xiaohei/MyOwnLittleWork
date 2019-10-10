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
		self.train = True
		if path.split('/')[-1] == 'ts':
			self.train = False
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
			label_res = pad_sequence([x[2] for x in tmps], batch_first=True)	# [B, T, C, num_bins]
			label_res = label_res.permute(2, 0, 1, 3)	# [C, B, T, num_bins]
			return x_res, vad_res, label_res
		else:
			mix = self.mixes[index]
			try:
				#     details of path       #
				x = np.load(os.path.join(self.root, mix), allow_pickle=True)
				vad_x = compute_vad_mask(x[0], self.vad_threshold, complex_=not self.train)

				#     details of label      #
				if self.train:
					label = x[1:]

					#     trans to t.Tensor     #
					mix_x = t.from_numpy(x[0])
					vad_x = t.from_numpy(vad_x)
					label = t.from_numpy(label)	# [C, T, num_bins]
					label = label.permute(1, 0, 2)	# [T, C, num_bins]
					return [mix_x, vad_x, label]
				else:
					label = x[1]	# [C, nsamples]
					return [x[0], vad_x, label]	# x[0] is complex [T, num_bins]

			except:
				new_index = random.randint(0, len(self) - 1)
				return self[new_index]

# Not use...
class Dataloader(object):
	def __init__(self, mix_speakers, batch_size=16, drop_last=False):
		self.mix_speakers = mix_speakers
		self.batch_size = batch_size
		self.drop_last = drop_last
		self.vad_threshold = vad_threshold

	def __len__(self):
		if self.drop_last or (len(self.mix_speakers) % self.batch_size) == 0:
			return len(self.mix_speakers) // self.batch_size
		else:
			return len(self.mix_speakers) // self.batch_size + 1

	def _total_length(self):
		return len(self.mix_speakers)

	def __iter__(self):
		l = self._total_length()
		for step in range(0, l, self.batch_size):
			if step + self.batch_size > l:
				if self.drop_last:
					break
				else:
					b = l - self.batch_size
			else:
				b = step
			mixes, vads, labels = self.mix_speakers[b:b + self.batch_size]

			yield [mixes, vads], labels

def get_bs(l, batch_size, index, drop_last=False):
	ans = [(s, index) for s in range(0, l, batch_size)]
	if ans[-1][0] + batch_size > l:
		if drop_last:
			ans = ans[:-1]
		else:
			ans[-1] = (l - batch_size, index)
	return ans

class DataLoader(object):
	def __init__(self, path, speaker_nums, batch_size=16, drop_last=False, vad_threshold=40, min_scale='batch'):
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

		self.mixes = [MixSpeakers(path.format(num=c), vad_threshold=vad_threshold) for c in speaker_nums]
		self.indices = []
		for ii, mix in enumerate(self.mixes):
			self.indices += get_bs(len(mix), batch_size, ii, drop_last)

		if min_scale == 'batch':
			random.shuffle(self.indeces)

	def __len__(self):
		return len(self.indices)

	def __iter__(self):
		for b, index in self.indices:
			if self.train:
				mixes, vads, labels = self.mixes[index][b:b + self.batch_size]
			else:
				mixes, vads, labels = self.mixes[index][b]

			yield [mixes, vads], labels