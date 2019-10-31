#coding=utf-8

import os
import random
import pickle
import numpy as np
import torch as t

from utils.process import compute_vad_mask
from torch.nn.utils.rnn import pack_sequence, pad_sequence

# data is saved as ... '.../data/Cspeakers/.../[tr, ts, cv]/[mix, s1, s2, ...]/*.wav'

class MixSpeakers(object):
	def __init__(self, path, samplerate=8000, duration=4, shuffle=True):
		'''
		samplerate * duration(s) = datalength
		path: .../Cspeakers/.../[tr, ...]/
		'''
		tmp_dir = os.listdir(path)
		self.C = len(tmp_dir) - 2
		self.mixes = os.listdir(os.path.join(path, mix))
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
		