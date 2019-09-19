import torch as t
import torch.nn as nn
import scipy.signal as signal
import numpy as np

from utils import util, process
from models import RSHNet
from itertools import permutations

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

def test_mixwav():
	path = '/Users/yuanzeyu/Desktop/test_wav'
	save_path = '/Users/yuanzeyu/Desktop/test_wav/saves'
	num_speakers = [2]
	snr_range = [-5., 5.]
	nums = [1, 1, 1]
	util.CreateMixWave(path, save_path, num_speakers, snr_range, nums, spl=44100)

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

if __name__ == '__main__':
	test_stft()