#coding=utf-8
import random
import glob
import os
import numpy as np
from scipy.io import wavfile

def ComputParameters(net, Mb=True):
	"""
	Return number parameters(not bytes) in nnet
	"""
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
		return random.uniform(snr_range)

	def NormalizeSignal(signal):
		signal = signal / (2 ** 15 - 1)
		s_p = np.sqrt(np.sum(signal ** 2))
		return signal / s_p if s_p > 0 else signal

	for num_spks in num_speakers:
		file_info = []
		for ii, data_type in data_types:
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
					sample, signal = wavfile.reaf(wav)
					assert sample == spl

					signal = NormalizeSignal(signal)
					if len(signal) < min_length:
						min_length = len(signal)

					sigs.append(signal)

				snrs[0] = 0.0
				save_name = save_name[:-1] + ".wav"
				file_info.append(txt_info + save_name)

				merge_wavs = np.zeros([num_spks + 1, min_length])
				for i in range(num_spks):
					random_b = random.randint(0, len(sigs[i] - min_length))
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
					save_dir = save_path + str(num_spks) + "speakers_0dB/wav8k/min/" + data_type + "/s" + str(i + 1) + "/"	# not sure...
					try:
						os.mkdir(save_dir)
					except BaseException:
						pass
					wavfile.write(save_dir + save_name, spl, merge_wavs[i].astype(np.int16))

				mix_dir = save_dir[:-3] + "mix/"
				try:
					os.mkdir(mix_dir)
				except BaseException:
					pass
				wavfile.write(mix_dir + wav_name, spl, merge_wavs[num_spks].astype(np.int16))

		
		with open(path + str(num_spks) + "speakers_" + "0dB/" + str(num_spks) + "speakers_" + "8k_0dB.txt", "w", encoding="utf-8") as f:
			for row in file_info:
				f.write(row)
				f.write("\n")