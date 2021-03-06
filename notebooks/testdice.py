import pickle
import pdb
from matplotlib import pyplot as plt

import setup

dices = [] #dices for label=1
losses= []
accuracies=[]
iterations=[]
test_dices=[]
test_accuracies=[]

i=                pickle.load(open(setup.MONITOR_FOLDER%"i.int",'r'))
dices=            pickle.load(open(setup.MONITOR_FOLDER%"dices.list",'r'))
losses=           pickle.load(open(setup.MONITOR_FOLDER%"losses.list",'r'))
accuracies=       pickle.load(open(setup.MONITOR_FOLDER%"accuracies.list",'r'))
iterations =      pickle.load(open(setup.MONITOR_FOLDER%"iterations.list",'r'))
test_dices =      pickle.load(open(setup.MONITOR_FOLDER%"test_dices.list",'r'))
test_accuracies = pickle.load(open(setup.MONITOR_FOLDER%"test_accuracies.list",'r'))

dice_iter = zip(test_dices,iterations)
dice_iter = sorted(dice_iter, key=lambda t:t[0], reverse=True)
for ji in range(10):
    print str(ji+1)+'th best test Dice:\t',round(dice_iter[ji][0],3),'\tAt iteration:\t',dice_iter[ji][1]

fig, axs = plt.subplots(2, 3, figsize=(9, 6))

axs[0][0].set_title('Train loss')
axs[0][0].plot(losses)

axs[0][1].set_title('Train dice')
axs[0][1].set_ylim([0,1])
axs[0][1].plot(dices)

axs[0][2].set_title('Test dice')
axs[0][2].set_ylim([0,1])
axs[0][2].plot(test_dices)

axs[1][1].set_title('Train acc')
axs[1][1].set_ylim([0.5,1])
axs[1][1].plot(accuracies)

axs[1][2].set_title('Test acc')
axs[1][2].set_ylim([0.5,1])
axs[1][2].plot(test_accuracies)

plt.show()