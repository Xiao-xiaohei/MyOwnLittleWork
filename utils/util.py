#coding=utf-8
import random
import glob
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

bug = True

def ComputParameters(net, Mb=True):
	'''
	Return number parameters(not bytes) in nnet
	'''
	ans = sum([param.nelement() for param in net.parameters()])
	return ans / 10**6 if Mb else ans

def CreateMixWave(path, save_path, num_speakers, snr_range, nums, spl=8000, reverberation=False):
	'''
		path: str, wav path, raw data
		num_speakers: list
		snr_range: [min, max]
		nums: list as [tr_num, cv_num, ts_num]
		reverberation: Not now
	'''
	tr_path = path + '/tr' # not sure...
	ts_path = path + '/ts'
	tr_cv = glob.glob(tr_path + '/*/*.wav')
	ts = glob.glob(ts_path + '/*/*.wav')
	random.shuffle(tr_cv)
	random.shuffle(ts)
	cut_num = int(len(tr_cv) * nums[0] / (nums[0] + nums[1]))
	tr = tr_cv[:cut_num]
	cv = tr_cv[cut_num:]
	wavs = [tr, cv, ts]
	data_types = ['tr', 'cv', 'ts']

	def GetSNR(snr_range):
		return random.uniform(*snr_range)

	def NormalizeSignal(signal):
		signal = signal / (2 ** 15 - 1)
		s_p = np.sqrt(np.sum(signal ** 2))
		return signal / s_p if s_p > 0 else signal

	for num_spks in num_speakers:
		file_info = []
		for ii, data_type in enumerate(data_types):
			wav_pairs = []
			tmp_num = 0
			type_num = nums[ii]
			while tmp_num < type_num:
				wav_pair = random.sample(wavs[ii], num_spks)
				wav_pair = sorted(wav_pair)
				if wav_pair not in wav_pairs:
					tmp_num += 1
					wav_pairs.append(wav_pair)

			for wav_pair in tqdm(wav_pairs):
				min_length = np.Inf
				snrs = []
				sigs = []
				save_name = ""
				txt_info = ""

				for wav in wav_pair:
					tmp_snr = GetSNR(snr_range)
					snrs.append(tmp_snr)
					wav_name = os.path.splitext(os.path.split(wav)[1])[0]
					save_name += (wav_name + "_" + str(tmp_snr) + "_")
					txt_info += (wav + " " + str(tmp_snr) + " ")
					sample, signal = wavfile.read(wav)
					assert sample == spl

					signal = NormalizeSignal(signal)
					if len(signal) < min_length:
						min_length = len(signal)

					sigs.append(signal[:, 0])

				snrs[0] = 0.0
				save_name = save_name[:-1] + ".wav"
				file_info.append(txt_info + save_name)

				merge_wavs = np.zeros([num_spks + 1, min_length])
				for i in range(num_spks):
					random_b = random.randint(0, len(sigs[i]) - min_length)
					merge_wavs[i, :] = 10 ** (snrs[i] / 20) * sigs[i][random_b:random_b + min_length]
				merge_wavs[num_spks, :] = np.sum(merge_wavs[:-1], axis=0)

				################################
				#     go for some fooooood     #
				#          2019.09.14          #
				################################

				max_amp = np.max(merge_wavs)
				merge_wavs /= max_amp
				merge_wavs *= 0.9
				merge_wavs = merge_wavs * (2 ** 15 - 1)

				save_dir = ""
				for i in range(num_spks):
					save_dir = save_path + str(num_spks) + "_" + data_type + "/s" + str(i + 1) + "/"	# not sure...
					try:
						os.makedirs(save_dir)
					except BaseException:
						pass
					wavfile.write(save_dir + save_name, spl, merge_wavs[i].astype(np.int16))

				mix_dir = save_dir[:-3] + "mix/"
				try:
					os.makedirs(mix_dir)
				except BaseException:
					pass
				wavfile.write(mix_dir + wav_name, spl, merge_wavs[num_spks].astype(np.int16))

		try:
			os.makedirs(path + str(num_spks) + "speakers_" + "0dB/")
		except BaseException:
			pass
		with open(path + str(num_spks) + "speakers_" + "0dB/" + str(num_spks) + "speakers_" + "8k_0dB.txt", "w", encoding="utf-8") as f:
			for row in file_info:
				f.write(row)
				f.write("\n")

def SegmentAxis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
	'''
	Organize overlapped(if it is) frames
	come from https://github.com/pchao6/LSTM_PIT_Speech_Separation/blob/master/utils.py
	'''
	if axis is None:
		a = np.ravel(a)
		axis = 0

	l = a.shape[axis]

	if overlap >= length:
		raise ValueError("Overlap mustn't longer than frame size!")
	if overlap < 0 or length < 0:
		raise ValueError("Overlap and length must be positive!")

	if l < length or (l - length) % (length - overlap):
		if l > length:
			roundup = length + (1 + (l - length) // (length - overlap)) * (
					length - overlap)
			rounddown = length + ((l - length) // (length - overlap)) * (
					length - overlap)
		else:
			roundup = length
			rounddown = 0
		assert rounddown < l < roundup
		assert roundup == rounddown + (length - overlap) or (
				roundup == length and rounddown == 0)
		a = a.swapaxes(-1, axis)

		if end == 'cut':
			a = a[..., :rounddown]
		elif end in ['pad', 'wrap']:  # copying will be necessary
			s = list(a.shape)
			s[-1] = roundup
			b = np.empty(s, dtype=a.dtype)
			b[..., :l] = a
			if end == 'pad':
				b[..., l:] = endvalue
			elif end == 'wrap':
				b[..., l:] = a[..., :roundup - l]
			a = b

		a = a.swapaxes(-1, axis)

	l = a.shape[axis]
	if l == 0:
		raise ValueError("Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'.")
	assert l >= length
	assert (l - length) % (length - overlap) == 0
	n == 1 + (l - length) // (length - overlap)
	s = a.strides[axis]
	newshape = a.shape[:axis] + (n, length) + a.shape[axis + 1:]
	newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]

	if not a.flags.contiguous:
		a = a.copy()
		newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
		return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)

	try:
		return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)
	except BaseException:
		warnings.warn("Problem with ndarray creation forces copy.")
		a = a.copy()
		# Shape doesn't change but strides does
		newstrides = a.strides[:axis] + ((length - overlap) * s, s) + a.strides[axis + 1:]
		return np.ndarray.__new__(np.ndarray, strides=newstrides, shape=newshape, buffer=a, dtype=a.dtype)