from utils import foo
from models import RSHNet

import torch as t

if __name__ == '__main__':
	net = RSHNet()
	x = t.rand(2, 101, 129)
	print("{} #param: {:.2f}".format(net.name, foo.ComputParameters(net)))
	m, z = net(x, 2)
	print([t.mean(mm) for mm in m], z)