# coding: utf-8

# In[6]:

# Most Code here
import sys
import scipy.misc, scipy.ndimage.interpolation
import pickle
sys.path.append("../data/layers")
sys.path.append("../")
sys.path.append("~/caffe/python/")
import os
import plyvel, saratan_utils, math, re, time
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
# plt.set_cmap('gray')
# get_ipython().magic(u'matplotlib inline')
from IPython import display
import pandas
from PIL import Image,ImageFilter

def dice(prediction, segmentation, label_of_interest = 1):
    """ Takes 2 2-D arrays with class labels, and return a float dice score.
    Only label=label_of_interest is considered """
    if prediction.shape != segmentation.shape:
        raise ValueError("Shape mismatch between given arrays. prediction %s vs segmentation %s"                          % (str(prediction.shape), str(segmentation.shape)))

    n_liver_seg = np.sum(segmentation==label_of_interest)
    n_liver_pred= np.sum(prediction == label_of_interest)
    denominator = n_liver_pred + n_liver_seg
    if denominator == 0:
        return -1

    liver_intersection   = np.logical_and(prediction==label_of_interest, segmentation==label_of_interest)
    n_liver_intersection = np.sum(liver_intersection)

    dice_score = 2.0*n_liver_intersection / denominator
    return dice_score
    
import caffe
print caffe.__file__

if config.CAFE_MODE is 'GPU':
    caffe.set_mode_gpu()
else if config.CAFE_MODE is 'CPU':
    caffe.set_mode_cpu()
else:
    raise NameError('Invalid CAFE_MODE')

# Load net
try : del solver 
except: pass
solver = caffe.SGDSolver("solver_unet.prototxt")
blobs = solver.net.blobs
params = solver.net.params

testblobs = solver.test_nets[0].blobs
testparams= solver.test_nets[0].params


onlyfiles = next(os.walk(config.TEST_STATE_FOLDER))[2] #dir is your directory path as string
totalfile = len(onlyfiles) / 2

iters = []
dices = []

for i in range(totalfile):
    iteration = (i+1)*500
    solver.net.copy_from(config.TEST_WEIGHT_FILE%iteration)
    solver.test_nets[0].copy_from(config.TEST_WEIGHT_FILE%iteration)
    # TMP : Test network on 200 slices
    print 'iter ', iteration
    print 'file: ', config.TEST_WEIGHT_FILE%iteration
    if True:
        tmp_dices = []
        neg_dice_count = 0
        for _ in range(300):
            solver.test_nets[0].forward()
            img_=testblobs['data'].data[0,0]
            seg_=testblobs['label'].data[0,0]
            dice_=dice(seg_,np.argmax(testblobs['score'].data[0],axis=0))
            if dice_ >= 0:
                tmp_dices.append(dice_)
            else:
                neg_dice_count += 1
            n_liver=np.sum(seg_>0)
            percent_liver = 100.0*n_liver / seg_.size
        print "Avg dice",np.average(tmp_dices)
        print "-1's :",neg_dice_count
        iters.append(iteration)
        dices.append(np.average(tmp_dices))

dice_iter = zip(dices,iters)
dice_iter = sorted(dice_iter, key=lambda t:t[0], reverse=True)
for ji in range(10):
    print str(ji+1)+'th best test Dice:\t',round(dice_iter[ji][0],3),'\tAt iteration:\t',dice_iter[ji][1]