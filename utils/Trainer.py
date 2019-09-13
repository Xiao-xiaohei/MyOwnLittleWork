import torch as t

from torchnet import meter
from tqdm import tqdm 

class Trainer(object):
	'''
	An abstract process for train.
	run()
	loss()
	'''
	def __init__(self, opt):
		self.vis = Visualizer(opt.env, port=opt.vis_port) if opt.vis else None
		self.model = getattr(models, opt.model_name)(**opt.model_kwargs)

		if opt.load_model_path:
			self.model.load_state_dict(t.load(opt.load_model_path))
		
		self.model.to(opt.device)
		self.opt = opt

	def set_optimizer(self):
		supported_optimizer = {
			"sgd": t.optim.SGD,	# momentum, weight_decay, lr
			"adam": t.optim.Adam,	# weight_decay, lr
			"rmsprop": t.optim.RMSprop 	# momentum, weight_decay, lr
		}
		if self.opt.optimizer not in supported_optimizer:
			raise ValueError("Unsupported optimizer {}"format(self.opt.optimizer))
		_kwargs = {
			"lr": self.opt.lr,
			"weight_decay": self.opt.weight_decay
		}
		if self.opt.optimizer != 'adam':
			_kwargs["momentum"] = self.opt.momentum
		self.optimizer = supported_optimizer[self.opt.optimizer](self.model.parameters(), **_kwargs)

	def loss(self, output, label):
		pass

	def run(self):
		self.set_optimizer()

		train_loss_meter = meter.AverageValueMeter()
		val_loss_meter = meter.AverageValueMeter()

		for epoch in range(self.opt.max_epoch):
			train_loss_meter.reset()
			val_loss_meter.reset()

			self.model.train()

			for ii, (data, label) in tqdm(enumerate(self.train_dataloader)):
				inputs = data.to(self.opt.device, dtype=t.float)
				target = label.to(self.opt.device)

				self.optimizer.zero_grad()
				output = self.model(inputs)
				loss = self.loss(output, target)
				loss.backward()
				self.optimizer.step()

				train_loss_meter.add(loss.item())

				if (ii + 1) % self.opt.print_freq == 0 and self.vis:
					vis.plot('train_loss', train_loss_meter.value()[0])

			self.model.eval()

			with t.no_grad():
				for ii, (data, label) in tqdm(enumerate(self.val_dataloader)):
					inputs = data.to(self.opt.device, dtype=t.float)
					target = label.to(self.opt.device)

					output = self.model(inputs)
					loss = self.loss(output, target)

					val_loss_meter.add(loss.item())
					if self.vis:
						vis.plot('val_loss', val_loss_meter.value()[0])

			save_path = os.path.join(self.checkpoint, "{}.{:d}.pth".format(self.model.name, epoch))
			t.save(self.model.state_dict(), save_path)