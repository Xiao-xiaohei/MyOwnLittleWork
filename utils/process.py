import string
import numpy as np
import scipy.io.wavfile as wf

from scipy import signal
from util import SegmentAxis
from numpy.fft import rfft, irfft

MAX_INT16 = np.iinfo(np.int16).max

def write_wav(fname, samps, fs=16000, normalize=True):
	'''
	Write wav files in int16, support single/multi-channel
	'''
	if normalize:
		samps = samps * MAX_INT16
	# scipy.io.wavfile.write could write single/multi-channel files
	# for multi-channel, accept ndarray [Nsamples, Nchannels]
	if samps.ndim != 1 and samps.shape[0] < samps.shape[1]:
		samps = np.transpose(samps)
		samps = np.squeeze(samps)
	# same as MATLAB and kaldi
	samps_int16 = samps.astype(np.int16)
	fdir = os.path.dirname(fname)
	if fdir and not os.path.exists(fdir):
		os.makedirs(fdir)
	# NOTE: librosa 0.6.0 seems could not write non-float narray
	#       so use scipy.io.wavfile instead
	wf.write(fname, fs, samps_int16)

def read_wav(fname, normalize=True, return_rate=False):
	'''
	Read wave files using scipy.io.wavfile(support multi-channel)
	'''
	# samps_int16: N x C or N
	#   N: number of samples
	#   C: number of channels
	samp_rate, samps_int16 = wf.read(fname)
	# N x C => C x N
	samps = samps_int16.astype(np.float)
	# tranpose because I used to put channel axis first
	if samps.ndim != 1:
		samps = np.transpose(samps)
	# normalize like MATLAB and librosa
	if normalize:
		samps = samps / MAX_INT16
	if return_rate:
		return samp_rate, samps
	return samps

def stft(signal, fft_size=1024, fft_shift=256, window=signal.blackman, padding=True, window_length=None):
	'''
	Compute Short time Fourier Transformation of a signal
	'''
	# signal: N x C or N
	#   N: number of samples
	#   C: number of channels
	
	if padding:
		pad = [(0, 0)] * signal.ndim
		pad[0] = [fft_size - fft_shift, fft_size - fft_shift]
		signal = np.pad(signal, pad, mode='constant')

	num_frames = np.ceil((signal.shape[0] - fft_size + fft_shift) / fft_shift).astype(np.int)
	padded_length = num_frames * fft_shift + fft_size - fft_shift
	pad = [(0, 0)] * signal.ndim
	pad[0] = [0, padded_length - signal.shape[0]]

	if window_length is None:
		window = window(fft_size)
	else:
		window = window(window_length)
		window = np.pad(window, (0, fft_size - window_length), mode='constant')

	signal_seg = SegmentAxis(signal, fft_size, fft_size - fft_shift, axis=0)
	
	letters = string.ascii_lowercase
	mapping = letters[:signal_seg.ndim] + ',' + letters[time_dim + 1] \
			  + '->' + letters[:time_signal_seg.ndim]

	return rfft(np.einsum(mapping, signal_seg, window),
				axis=time_dim + 1)