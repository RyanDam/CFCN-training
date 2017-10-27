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

testdices = [] #dices for label=1
iterations = []

testdices=pickle.load(open("test_dices.list",'r'))
iterations=pickle.load(open("iterations.list",'r'))

plt.plot(testdices)
plt.show()

dice_iter = zip(testdices,iterations)
dice_iter = sorted(dice_iter, key=lambda t:t[0], reverse=True)
for ji in range(10):
    print str(ji+1)+'th best test Dice:\t',round(dice_iter[ji][0],3),'\tAt iteration:\t',dice_iter[ji][1]

pdb.set_trace()

print 1