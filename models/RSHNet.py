import torch as t
import torch.nn as nn

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

class RSHNet(nn.Module):
	'''
		Recurrent Selective Hearing Networks

		Reference:
			Kinoshita K, Drude L, Delcroix M, et al. Listening to Each Speaker One by One with Recurrent Selective Hearing Networks[C]. international conference on acoustics, speech, and signal processing, 2018: 5064-5068.
	'''
	def __init__(self, num_bins=129,
			rnn="lstm",
			num_layers=2,
			hidden_size=600,
			act_func="sigmoid",
			bidirectional=True,
			greedy=False):
		super(RSHNet, self).__init__()
		self.name = "RSHNet"
		self.greedy = greedy
		if act_func not in ["tanh", "sigmoid"]:
			raise ValueError("Unsupported activation function type:{}".format(act_func))

		rnn = rnn.upper()
		if rnn not in ["RNN", "GRU", "LSTM"]:
			raise ValueError("Unsupported rnn type:{}".format(rnn))

		self.rnn = getattr(nn, rnn)(
			num_bins * 2,
			hidden_size,
			num_layers,
			batch_first=True,
			bidirectional=bidirectional
		)

		self.mask = nn.Linear(
			hidden_size * 2 if bidirectional else hidden_size,
			num_bins
		)

		self.act_func = {
			"sigmoid":t.sigmoid,
			"tanh":t.tanh}[act_func]

		self.flag = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)

	def forward(self, x):
		'''
		input:
			x: [B, T, num_bins x 2]
		output:
			Mask: [B, T, num_bins]
			flag: [B]
		'''
		is_packed = isinstance(x, PackedSequence)
		if not is_packed and x.dim() != 3:
			x = t.unsqueeze(x, 0)

		x, _ = self.rnn(x)
		if is_packed:
			x, _ = pad_packed_sequence(x, batch_first=True)
		m = self.mask(x)
		m = self.act_func(m)
		z = self.flag(x[:, -1, :])	# [B, 1]
		#z = t.mean(t.sigmoid(z).squeeze(2), 1)
		return m, t.squeeze(z)

	"""
	def forward(self, x, C):
		'''
			input:
				x: [B, T, num_bins] concate with M [B, T, num_bins] 
					label_M (if greedy: [C, B, T, num_bins]
				C: scalar
			output:
				M_: [C, B, T, num_bins]
				res_M: [B, T, num_bins]
				flags: z [B, C]
		'''
		if x.dim() < 3:
			x = t.unsqueeze(x, 0)

		if self.greedy and x.dim() < 4:	# x: [C + 1, B, T, num_bins] or [1, B, T, num_bins]
			raise ValueError("Lost label_M for loss computation!")
		
		if self.greedy:
			label_M = x[1:]
			x = x[0]
			res = t.ones([C, x.shape[0]])	# to mask res loss
			g_loss = []
		Ms = []
		zs = []
		M = t.ones(x.shape)

		greedy_loss = None

		for i in range(C):
			y = t.cat([x, M], dim=-1)
			y, _ = self.rnn(y)	# y: [B, T, hidden_size * 2]
			m = self.mask(y) # m: [B, T, num_bins]
			m = self.act_func(m)
			if self.greedy:
				tmp_M = t.stack([m for _ in range(C)])
				tmp_loss = t.norm(tmp_M - label_M, p='fro', dim=[-2, -1])	# size: [C, B]
				# weight/mask it
				tmp_loss += (t.max(tmp_loss) * t.ones(tmp_loss.shape))
				# get indices
				indicate = t.min(tmp_loss, dim=0)	# both value: [B,] and index: [B, ]
				tmp_index = indicate.indices
				out_loss = indicate.values
				new_mask = []
				B = x.shape[0]
				for iii in range(B):
					new_mask.append(label_M[tmp_index[iii]][iii])
					res[tmp_index[iii]][iii] = 0

				m = t.stack(new_mask)	# [B, T, num_bins]
				g_loss.append(out_loss)
				if i == C - 1:
					greedy_loss = t.stack(g_loss)	# [C, B]

			Ms.append(m.unsqueeze(0))
			M -= m
			z = self.flag(y)
			z = t.mean(t.sigmoid(z).squeeze(2), 1)
			zs.append(z.unsqueeze(0))

		return t.cat(Ms, dim=0), M, t.cat(zs, dim=0).permute(1, 0), greedy_loss
	"""