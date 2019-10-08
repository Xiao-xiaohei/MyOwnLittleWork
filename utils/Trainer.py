import torch as t
import models

from torchnet import meter
from tqdm import tqdm
from utils.visualize import Visualizer

class Trainer(object):
	'''
	An abstract process for train.
	run()
	recursive_loss()
	'''
	def __init__(self, opt):
		self.vis = Visualizer(opt.env, port=opt.vis_port) if opt.vis else None
		self.model = getattr(models, opt.model)(**opt.model_kwargs)

		if opt.load_model_path:
			self.model.load_state_dict(t.load(opt.load_model_path))
		
		self.model.to(opt.device)
		self.opt = opt

	def set_train_dataloader(self):
		'''
		return train_dataloader & cv_dataloader
		in the future, the func will be more abstract ! But not now... dbq....
		'''
		pass

	def set_test_dataloader(self):
		'''
		return tesr_dataloader
		'''
		pass

	def compute_evaluation(self, datas, label, types):
		'''
		return evaluation, types: eg Acc, SDR...
		'''
		pass

	def set_optimizer(self):
		supported_optimizer = {
			"sgd": t.optim.SGD,	# momentum, weight_decay, lr
			"adam": t.optim.Adam,	# weight_decay, lr
			"rmsprop": t.optim.RMSprop 	# momentum, weight_decay, lr
		}
		if self.opt.optimizer not in supported_optimizer:
			raise ValueError("Unsupported optimizer {}".format(self.opt.optimizer))
		_kwargs = {
			"lr": self.opt.lr,
			"weight_decay": self.opt.weight_decay
		}
		if self.opt.optimizer != 'adam':
			_kwargs["momentum"] = self.opt.momentum
		self.optimizer = supported_optimizer[self.opt.optimizer](self.model.parameters(), **_kwargs)

	def recursive_loss(self, data, target):
		pass

	def run(self):
		self.set_optimizer()

		train_dataloader, cv_dataloader = self.set_train_dataloader()
		train_loss_meter = meter.AverageValueMeter()
		val_loss_meter = meter.AverageValueMeter()

		for epoch in range(self.opt.max_epoch):
			train_loss_meter.reset()
			val_loss_meter.reset()

			self.model.train()

			for ii, (data, label) in tqdm(enumerate(train_dataloader)):
				if isinstance(data, list):
					inputs = [d.to(self.opt.device, dtype=t.float32) for d in data]
				else:
					inputs = data.to(self.opt.device, dtype=t.float32)
				target = label.to(self.opt.device)

				self.optimizer.zero_grad()

				# output = self.model(inputs)

				loss = self.recursive_loss(inputs, target)
				
				loss.backward()
				self.optimizer.step()

				train_loss_meter.add(loss.item())

				if (ii + 1) % self.opt.print_freq == 0 and self.vis:
					vis.plot('train_loss', train_loss_meter.value()[0])

			self.model.eval()

			with t.no_grad():
				for ii, (data, label) in tqdm(enumerate(cv_dataloader)):
					if isinstance(data, list):
						inputs = [d.to(self.opt.device, dtype=t.float32) for d in data]
					else:
						inputs = data.to(self.opt.device, dtype=t.float32)
					target = label.to(self.opt.device, dtype=t.float32)

					output = self.model(inputs)
					loss = self.loss(output, target)

					val_loss_meter.add(loss.item())
					if self.vis:
						vis.plot('val_loss', val_loss_meter.value()[0])

			save_path = os.path.join(self.checkpoint, "{}.{:d}.pth".format(self.model.name, epoch))
			t.save(self.model.state_dict(), save_path)

	def test(self):
		ts_dataloader = self.set_test_dataloader()

		self.model.eval()

		for ii, (data, label) in tqdm(enumerate(ts_dataloader)):
			if isinstance(data, list):
				inputs = [d.to(self.opt.device, dtype=t.float32) for d in data]
			else:
				inputs = data.to(self.opt.device, dtype=t.float32)
			# target = label.to(self.opt.device, dtype=t.float32)

			ans = self.compute_evaluation(inputs, target, self.opt.evaluations)	# ans eg dic{'Acc':True/False, 'SDR':xxx, ...}