import torch as t
import torch.nn as nn
import scipy.signal as signal
import numpy as np

from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from utils import util, process
from models import RSHNet
from itertools import permutations
from main import RSHNetTrainer
from config import opt

from data import MixSpeakers

def loss(output, label):
	'''
		output:
			M [C, B, T, num_bins]
			res_M [B, T, num_bins]
			and
			z [B, C]
		label:
			M [C, B, T, num_bins]
			z [B, C]

		compute loss using PIT for M ~ M
		L_{Mask}: MSE?
		L_{flag}: CrossEntropy? MSE?
		L_{res_Mask}: Square？ or CrossEntropy that minus is heavily gg!
	'''
	output_flags = output[2]	# [B, C]
	L_flag = nn.BCELoss()(output_flags, label[1])

	L_resMask = t.norm(output[1], 2)
	M = output[0]
	C = M.shape[0]
	# pit_mat with shape [C!, B]
	pit_mat = t.stack([mse_loss(M, label[0], p) for p in permutations(range(C))])
	L_mask, min_per = t.min(pit_mat, dim=0)
	return L_mask + L_flag + L_resMask

def mse_loss(obtain_m, ref_m, permutation):
	'''
		obtain_m: the elimated masks [C, B, T, num_bins]
		ref_m: the reference masks [C, B, T, num_bins]
		permutation: one permutation of C!
	'''
	# get a loss with shape [B, ]
	return sum([mse(obtain_m[s], ref_m[t]) for s, t in enumerate(permutation)])

def mse(ob_m, ref_m):
	'''
		input:
			ob_m: [B, T, num_bins]
			ref_m: [B, T, num_bins]
		out:
			loss: [B, ]
	'''
	return t.norm(ob_m - ref_m, p='fro', dim=[-2, -1])

def test_net():
	net = RSHNet()
	x = [t.rand(100, 129), t.rand(90, 129), t.rand(80, 129)]
	xx = pack_sequence(x)
	xxx, batch_length = pad_packed_sequence(xx, batch_first=True)
	M = t.ones(xxx.shape)
	xxxx = t.cat([xxx, M], dim=-1)
	print(xxxx.shape)
	xxxxx = pack_padded_sequence(xxxx, batch_length, batch_first=True)
	#print("{} #param: {:.2f}".format(net.name, util.ComputParameters(net)))
	m, z = net(xxxxx)
	print(m.shape, z.shape)

def test_recursive_loss(**kwargs):
	opt._parse(kwargs)
	trainer = RSHNetTrainer(opt)
	data = t.rand(10, 100, 129)
	label = t.rand(3, 10, 100, 129)
	L1, L2, L3 = trainer.recursive_loss(data, label)
	print(L1, L2, L3)
	
'''
def test_net():
	net = RSHNet()
	x = t.rand(2, 101, 129)
	print("{} #param: {:.2f}".format(net.name, util.ComputParameters(net)))
	m, resm, z, _ = net(x, 3)
	print(m.shape, resm.shape, z.shape)

	C = 3
	ref_m = t.rand([3, 2, 101, 129])
	ref_z = t.rand([2, 3])
	label = [ref_m, ref_z]
	output = [m, resm, z]
	Loss = loss(output, label)
	print(Loss)

	x = t.rand(4, 2, 101, 129)	# [C + 1, B, T, num_bins]
	net.greedy = True
	m, resm, z, _ = net(x, 3)
	print(m.shape, resm.shape, z.shape)

	C = 3
	ref_m = t.rand([3, 2, 101, 129])
	ref_z = t.rand([2, 3])
	label = [ref_m, ref_z]
	output = [m, resm, z]
	Loss = loss(output, label)
	print(Loss)
'''

def test_mixwav():
	path = '/Users/yuanzeyu/Desktop/test_wav'
	save_path = '/Users/yuanzeyu/Desktop/test_wav/saves'
	num_speakers = [2]
	snr_range = [-5., 5.]
	nums = [1, 1, 1]
	util.CreateMixWave(path, save_path, num_speakers, snr_range, nums, spl=44100)

def test_createlabels():
	speaker_nums = [2]
	data_path = '/Users/yuanzeyu/Desktop/test_wav/saves'
	save_path = '/Users/yuanzeyu/Desktop/test_wav/npy_saves'
	window_size = 1024
	window_shift = 768
	spl = 44100
	util.CreateLabelsAll(speaker_nums, data_path, save_path, window_size, window_shift, spl)

def test_stft():
	path = '/Users/yuanzeyu/Desktop/mix_LSHNY_-5db.wav'
	sig = process.read_wav(path)
	sig = np.asarray([sig, sig])
	print(sig.shape)
	#stft_sig = process.stft(sig)
	#print(stft_sig.shape)
	stft_sig_ = signal.stft(sig, nperseg=1024, noverlap=768, nfft=1024, window='blackman')
	print(stft_sig_[0].shape, stft_sig_[1].shape, stft_sig_[2].shape)
	#print('##############')
	#stft_sig = stft_sig[:-1, :]
	#print(np.sum(stft_sig.T - stft_sig_[2][0]))

def test_MixSpeakers():
	path = '/Users/yuanzeyu/Desktop/test_wav/npy_saves/2speakers/tr'
	mixes = MixSpeakers(path)
	res = mixes[0:1]
	print(res[0][0].shape)

if __name__ == '__main__':
	test_net()