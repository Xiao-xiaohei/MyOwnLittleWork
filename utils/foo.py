def ComputParameters(net, Mb=True):
	"""
	Return number parameters(not bytes) in nnet
	"""
	ans = sum([param.nelement() for param in net.parameters()])
	return ans / 10**6 if Mb else ans