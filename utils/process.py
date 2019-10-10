import string
import numpy as np
import scipy.io.wavfile as wf

from scipy import signal
from numpy.fft import rfft, irfft

MAX_INT16 = np.iinfo(np.int16).max

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
	n = 1 + (l - length) // (length - overlap)
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
	mapping = letters[:signal_seg.ndim] + ',' + letters[0 + 1] \
			  + '->' + letters[:signal_seg.ndim]

	return rfft(np.einsum(mapping, signal_seg, window),
				axis=0 + 1)

def compute_vad_mask(spec, threshold_db, apply_exp=True, complex_=False):
	if complex_:
		spec = np.abs(spec)
	if apply_exp:
		spec = np.exp(spec)
	spec_db = 20 * np.log10(spec)
	max_db = np.max(spec_db)
	threshold = 10 ** ((max_db - threshold_db) / 20)
	mask = np.array(spec > threshold, dtype=np.float32)
	return mask