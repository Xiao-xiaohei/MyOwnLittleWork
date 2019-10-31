import numpy as np
import matplotlib.pyplot as plt

acc = [None, None, None]

epoch = 0
mask = 'psm_'
normalize = 'not_normalization_T_bins_'

path = './RSHNet.{epoch}.pth_{mask}{normalize}{spk}.npy'

for i in range(3):
	acc[i] = np.load(path.format(epoch=epoch, mask=mask, normalize=normalize, spk=i+2))
#acc[0] = np.load('./RSHNet.4.pth_asm_not_nomalize_checkpoint_2.npy')
#acc[1] = np.load('./RSHNet.4.pth_asm_not_nomalize_checkpoint_3.npy')
#acc[2] = np.load('./RSHNet.4.pth_asm_not_nomalize_checkpoint_4.npy')

'''
plt.plot(acc[0][:, 0])
plt.plot(acc[1][:, 0])
plt.plot(acc[2][:, 0])
plt.show()

plt.plot(acc[0][:, 1])
plt.plot(acc[1][:, 1])
plt.plot(acc[2][:, 1])
plt.show()


plt.plot(acc[1][:, 2])
plt.plot(acc[2][:, 2])
plt.show()

plt.plot(acc[2][:, 3])
plt.show()

for i in range(3):
	print("The last flag's range: max: {ma}, min: {mi}".format(ma=np.max(acc[i][:, -1]), mi=np.min(acc[i][:, -1])))
'''

max_threshold = 1.0
min_threshold = 0.0
delta = (max_threshold - min_threshold) / 100
thresholds = []
Accs = [[], [], [], []]
for ii in range(101):
	threshold = min_threshold + ii * delta
	thresholds.append(threshold)
	ACC = 0
	for i in range(3):
		l = acc[i].shape[0]
		C = acc[i].shape[1]
		Acc = 0
		for j in range(l):
			c = 0
			while(c < C):
				if acc[i][j][c] < threshold:
					c += 1
				else:
					break
			if c == C - 1:
				Acc += 1
		ac = Acc / l
		ACC += Acc
		Accs[i].append(ac)
		#print("{cc} speakers Acc: {acc}, above_Acc: {a_acc} and below_acc: {b_acc}".format(cc = C, acc=ac, a_acc=a_ac, b_acc=1-ac-a_ac))
	Accs[3].append(ACC/(l * 3))

for i in range(4):
	plt.plot(Accs[i])
max_ac = np.max(Accs[3])
threshold = thresholds[np.argmax(Accs[3])]
plt.savefig('{epoch}_{mask}{normalize}{max_ac}_with_{threshold}.png'.format(epoch=epoch, mask=mask, normalize=normalize, max_ac=max_ac, threshold=threshold))
plt.show()
