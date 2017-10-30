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

i=                pickle.load(open("monitor/i.int",'r'))
dices=            pickle.load(open("monitor/dices.list",'r'))
dices_2=          pickle.load(open("monitor/dices_2.list",'r'))
test_dices_2 =    pickle.load(open("monitor/test_dices_2.list",'r'))
losses=           pickle.load(open("monitor/losses.list",'r'))
accuracies=       pickle.load(open("monitor/accuracies.list",'r'))
iterations =      pickle.load(open("monitor/iterations.list",'r'))
test_dices =      pickle.load(open("monitor/test_dices.list",'r'))
test_accuracies = pickle.load(open("monitor/test_accuracies.list",'r'))

plt.plot(test_dices)
plt.show()

dice_iter = zip(test_dices,iterations)
dice_iter = sorted(dice_iter, key=lambda t:t[0], reverse=True)
for ji in range(10):
    print str(ji+1)+'th best test Dice:\t',round(dice_iter[ji][0],3),'\tAt iteration:\t',dice_iter[ji][1]

pdb.set_trace()

print 1