from utils import util
from models import RSHNet

import torch as t

if __name__ == '__main__':
	net = RSHNet()
	x = t.rand(2, 101, 129)
	print("{} #param: {:.2f}".format(net.name, util.ComputParameters(net)))
	m, resm, z = net(x, 2)
	print(m.shape, resm.shape, z.shape)