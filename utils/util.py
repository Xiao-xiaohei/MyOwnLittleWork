#coding=utf-8
import random
import glob
import os
import numpy as np
import scipy.signal as signal
from tqdm import tqdm
from scipy.io import wavfile
from .process import read_wav

bug = True

def ComputParameters(net, Mb=True):
	'''
	Return number parameters(not bytes) in nnet
	'''
	ans = sum([param.nelement() for param in net.parameters()])
	return ans / 10**6 if Mb else ans

def CreateMixWave(path, save_path, num_speakers, snr_range, nums, spl=8000, reverberation=False):
	'''
		path: str, wav path, raw data, eg: ../data and its sons are ../data/tr/sp_num/xx.wav & ../data/ts/sp_num/xx.wav
		num_speakers: list, eg: [2, 3, 4, ...]
		snr_range: [min, max]
		nums: list as [tr_num, cv_num, ts_num] eg: [20000, 5000, 3000]
		reverberation: Not now

		There will be:
			../save_data/Cspeakers/[tr.txt, ts.txt, cv.txt, tr/[mix, s1, s2, ...], ts/..., cv/...]
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
		for ii, data_type in enumerate(data_types):
			file_info = []

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

				#     go for some fooooood     #
				#          2019.09.14          #

				max_amp = np.max(merge_wavs)
				merge_wavs /= max_amp
				merge_wavs *= 0.9
				merge_wavs = merge_wavs * (2 ** 15 - 1)

				save_dir = ""
				for i in range(num_spks):
					save_dir = save_path + "/" + str(num_spks) + "speakers/" + data_type + "/s" + str(i + 1) + "/"
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
				wavfile.write(mix_dir + save_name, spl, merge_wavs[num_spks].astype(np.int16))

			try:
				os.makedirs(save_path + "/" + str(num_spks) + "speakers/")
			except BaseException:
				pass
			with open(save_path + "/" + str(num_spks) + "speakers/" + data_type + ".txt", "w", encoding="utf-8") as f:
				for row in file_info:
					f.write(row)
					f.write("\n")

def ComputeMasks(mix, cleans, mask_type='PSM'):
	'''
	mix: [frames, fft//2 + 1]	Complex Domain
	cleans: [s1_spec, s2_spec, ...]
	mask_type:	['PSM', 'IBM', 'IRM'...]
	'''
	if mask_type not in ['PSM', 'IBM', 'IRM']:
		raise ValueError("Unsupported mask type!")

	C = len(cleans)
	
	mix_abs = np.abs(mix)
	mix_angle = np.angle(mix)

	clean_abs = [np.abs(stft) for stft in cleans]
	clean_angle = [np.angle(stft) for stft in cleans]

	#inputs = np.concatenate((mix_abs, mix_angle), axis=1)
	
	if mask_type == 'PSM':
		cross_masks = [s_abs * np.cos(mix_angle - s_angle) for s_abs, s_angle in zip(clean_abs, clean_angle)]

	return mix_abs, mix_angle, cross_masks

def CreateLabelOnce(data, wav_path, data_type, save_path, window_size, window_shift, spl=8000):
	'''
	The 'wavpath' is the wavs of s1, s2, ..., sC and s_mix or other info to know which wavs are in the same group
	Here is the latter form from [info].txt created before.
	wav_path: [s1, snr1, s2, snr2, ..., mix]
	save_path: ../data --> ../data/Cspeakers/data_type/xxx.npy
	'''
	wav_path = wav_path.split()
	#This part to parse 'wav_path'
	wav_name = wav_path[-1]

	C = (len(wav_path) - 1) // 2
	data = data + "/" + str(C) + "speakers/" + data_type + "/"
	mix_wav = read_wav(data + "mix/" + wav_name)
	wavs = [read_wav(data + "s" + str(i + 1) + "/" + wav_name) for i in range(C)]
	new_name = None

	# Here maybe need check dim whether [C, N] #
	#         have done it in read_wav         #

	# That's pchao's method...
	# mix_stft = stft(mix_wav)
	# stfts = [stft(wav_signal) for wav_signal in wavs]

	def _stft(sig, fs=spl, nperseg=window_size, noverlap=window_size-window_shift, nfft=window_size, window='blackman'):
		_, _, ans = signal.stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window=window)
		return np.transpose(ans)

	# Here use scipy.signal.stft
	mix_stft = _stft(mix_wav)
	stfts = [_stft(wav_signal) for wav_signal in wavs]

	# Mask Computation eg IBM, IRM, PSM(mainly focused)...
	# go for dinner! --09.19 16:25

	inputs_abs, inputs_angle, labels = ComputeMasks(mix_stft, stfts)

	#  Save features...names' parse problem!  #
	#    or just process wav in dataloader?   #
	new_dir = str(C) + "speakers/" + data_type + "/"
	try:
		os.makedirs(save_path + "/" + new_dir)
	except BaseException:
		pass
	new_name = wav_name[:-3] + "npy"
	res = np.stack([inputs_abs] + labels, axis=0)
	np.save(save_path + "/" + new_dir + "/" + new_name, res)

	return

def CreateLabelsAll(speaker_nums, data_path, save_path, window_size, window_shift, spl=8000):
	'''
	preprocess all data, which includes num_speakers[2, 3, 4...] & ['tr', 'ts', 'cv']
	I make the hypothesis that the space to save data is enough...
	'''

	data_types = ['tr', 'cv', 'ts']

	# path problem

	for data_type in data_types:
		for spk_num in speaker_nums:
			parse_file = data_path + "/" + str(spk_num) + "speakers/" + data_type + ".txt"

			with open(parse_file, 'r') as f:
				info_lines = f.readlines()

				for line in info_lines:
					CreateLabelOnce(data_path, line, data_type, save_path, window_size, window_shift, spl)