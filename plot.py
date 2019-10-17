import numpy as np
import matplotlib.pyplot as plt

threshold = 0.5

acc = [None, None, None]

acc[0] = np.load('./psm_RSHNet.1.pth_asm_not_nomalize_checkpoint_2.npy')
acc[1] = np.load('./psm_RSHNet.1.pth_asm_not_nomalize_checkpoint_3.npy')
acc[2] = np.load('./psm_RSHNet.1.pth_asm_not_nomalize_checkpoint_4.npy')

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
	l = acc[i].shape[0]
	C = acc[i].shape[1]
	Acc = []
	Acc_above = []
	for j in range(l):
		c = 0
		while(c < C):
			if acc[i][j][c] < threshold:
				c += 1
			else:
				break
		if c == C - 1:
			Acc.append(1)
			Acc_above.append(0)
		elif c == C:
			Acc.append(0)
			Acc_above.append(1)
	ac = sum(Acc) / l
	a_ac = sum(Acc_above) / l
	print("{cc} speakers Acc: {acc}, above_Acc: {a_acc} and below_acc: {b_acc}".format(cc = C, acc=ac, a_acc=a_ac, b_acc=1-ac-a_ac))
