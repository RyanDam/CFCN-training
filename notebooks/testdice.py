import pickle
import pdb
from matplotlib import pyplot as plt

def smooth_last_n(arr, n=5, ignore=None):
    """Replaces the last n elements in arr (list) with their average."""
    subarr = np.array(arr[-n:])
    if ignore != None:
        subarr = subarr[subarr != ignore] 
    mean = np.mean(subarr)
    return arr[:-n]+[mean]

dices = [] #dices for label=1
dices_2 = [] #dices for label=2
losses= []
accuracies=[]
iterations=[]
test_dices=[]
test_dices_2=[]
test_accuracies=[]

PREFIX_FOLDER = 'monitor'

i=                pickle.load(open("%s/i.int"%PREFIX_FOLDER,'r'))
dices=            pickle.load(open("%s/dices.list"%PREFIX_FOLDER,'r'))
dices_2=          pickle.load(open("%s/dices_2.list"%PREFIX_FOLDER,'r'))
test_dices_2 =    pickle.load(open("%s/test_dices_2.list"%PREFIX_FOLDER,'r'))
losses=           pickle.load(open("%s/losses.list"%PREFIX_FOLDER,'r'))
accuracies=       pickle.load(open("%s/accuracies.list"%PREFIX_FOLDER,'r'))
iterations =      pickle.load(open("%s/iterations.list"%PREFIX_FOLDER,'r'))
test_dices =      pickle.load(open("%s/test_dices.list"%PREFIX_FOLDER,'r'))
test_accuracies = pickle.load(open("%s/test_accuracies.list"%PREFIX_FOLDER,'r'))

dice_iter = zip(test_dices,iterations)
dice_iter = sorted(dice_iter, key=lambda t:t[0], reverse=True)
for ji in range(10):
    print str(ji+1)+'th best test Dice:\t',round(dice_iter[ji][0],3),'\tAt iteration:\t',dice_iter[ji][1]

fig, axs = plt.subplots(2, 3, figsize=(9, 6))

axs[0][0].set_title('Train loss')
axs[0][0].plot(losses)

axs[0][1].set_title('Train dice')
axs[0][1].plot(dices)

axs[0][2].set_title('Test dice')
axs[0][2].plot(test_dices)

axs[1][1].set_title('Train acc')
axs[1][1].plot(accuracies)

axs[1][2].set_title('Test acc')
axs[1][2].plot(test_accuracies)

plt.show()