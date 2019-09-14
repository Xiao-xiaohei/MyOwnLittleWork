def ComputParameters(net, Mb=True):
	"""
	Return number parameters(not bytes) in nnet
	"""
	ans = sum([param.nelement() for param in net.parameters()])
	return ans / 10**6 if Mb else ans

def CreateMixWave(path, num_speakers, snr_range, reverberation=False):
	'''
		path: str, wav path, raw data
		num_speakers: int
		snr_range: [min, max]
		reverberation: Not now
	'''
	pass