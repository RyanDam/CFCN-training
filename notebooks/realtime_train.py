
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
from IPython import display
import pandas
from PIL import Image,ImageFilter

def hist(arr):
    """Print number of pixels for each label in the given image (arr)"""
    return "%.3f , %.3f , %.3f, %.3f" % (np.sum(arr==0),np.sum(arr==1),np.sum(arr==2),np.sum(arr==4))

def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage: 
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    axis_enabled = kwargs.get('axis',True)

    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        if not axis_enabled:
            plt.axis('off')
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        plt.figure(figsize=(n*5,10))
        for i in range(n):
            plt.subplot(1,n,i+1)
            plt.title(title[i])
            if not axis_enabled:
                plt.axis('off')
            plt.imshow(args[i], cmap[i], interpolation='none')
    plt.show()
        
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

    
def protobinary_to_array(filename, outpng=None):
    """ Filename is path to protobinary
    outpng is path to output png"""
    with open(filename,'r') as f:
        data = f.read()

    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob)) #returns shape (1,1,W,H)
    arr = arr[0,0,:,:] #ignore first 2 dimensions
    return  arr

dices_liver = []
dices_lesion= []
def predict(net, img, seg, meanimg):
    """Predicts an img using the trained net, and compares it to the label image (seg)"""
    net.blobs['data'].data[0]=(img-meanimg)
    prob=net.forward()['prob'][0]
    prediction = np.argmax(prob,axis=0)
    dice_liver = dice(prediction,seg,label_of_interest=1)
    dice_lesion = dice(prediction,seg,label_of_interest=2)
    dices_liver.append(dice_liver)
    dices_lesion.append(dice_lesion)
    print "Dice Liver:", dice_liver
    print "Dice Lesion:",dice_lesion
    print "Prediction class histogram",hist(prediction)
    print "Ground truth class histogram",hist(seg)
    plt.figure(figsize=(20,24))
    plt.subplot(1,3,1); plt.title("Image")
    plt.imshow(img)
    plt.subplot(1,3,2); plt.title("Ground truth")
    plt.imshow(seg)
    plt.subplot(1,3,3); plt.title("Prediction")
    plt.imshow(prediction)
    plt.show()
    
def read_imgs(dbimgit, dbsegit, n=1, print_keys=True):
    """Read img and label after skipping n keys in leveldb. Takes db iterators"""
    for _ in range(n):
        k1,vimg = dbimgit.next()
        k2,vseg = dbsegit.next()
    if print_keys:
        print "Keys:",k1,k2
    img=lutils.to_numpy_matrix(vimg)
    seg=lutils.to_numpy_matrix(vseg)
    return img,seg

def show_kernels(layer_blob_data, fast = False):
    """ Takes solver.net.params['conv1'][0].data and visualize the first channel of all kernels.
    If fast = False : subplots will be used, allowing to see each filter individually, but takes time.
    If fast = True : all filters are plotted in one image"""
    #Input has 4 dims, we only visualize 1st channel of each kernel 
    # (the conv weights that acts on the 1st channel of the input)
    data = layer_blob_data[:,0,:,:]
    if fast:
        raise NotImplementedError("todo")
    
    # Sort
    sorted_data = sorted(data, key=lambda x: np.sum(x))
    data = np.array(sorted_data)
    
    n_kernels = np.array(data).shape[0]
    plot_cols = 20 #number of images in one row
    plot_rows = math.ceil(n_kernels*1.0 / plot_cols)
    # Adjust figure plot size
    plt.figure(figsize=(min(plot_cols, n_kernels)*0.7, plot_rows*0.7))
    # Plot !
    vmin = np.min(data)
    vmax = np.max(data)
    print vmin,vmax
    for i in range(n_kernels):
        plt.subplot(plot_rows, plot_cols, i+1)
        plt.imshow(data[i], interpolation='none', vmin=vmin, vmax=vmax)
        plt.axis("off")
    plt.show()
        
def plot_deepliver_log(fname):
    """Takes file handle of deepliver log, and plots the 4 plots :
    Loss, avgAccuracy, avgJaccard, avgRecall"""
    f = open(fname, 'r')
    logs = f.read()
    plt.figure(figsize=(10,10))
    # Get iterations
    iterations = re.findall("Iteration (\d+), loss",logs)

    # Get&plot loss
    loss = zip(*re.findall("Iteration \d+, loss = ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)",logs))[0]
    length = min(len(iterations), len(loss))
    iterations_trunc, loss_trunc = iterations[:length], loss[:length]
    plt.plot(iterations,loss,label='Loss')
    #plt.show()
    #Get&plot metrics
    metrics = ['Accuracy','Recall','Jaccard']
    data = defaultdict(list) # data.keys() = metrics , data[metrics[0]] = list of values
    for i,metric in enumerate(metrics):
        regex = "Train net output #"+str(i)+": accuracy = ([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
        for result in re.findall(regex,logs):
            data[metric].append(result[0])

    for metric in data.keys():
        length = min(len(iterations),len(data[metric]))
        iterations_trunc, data_trunc = iterations[:length], data[metric][:length]
        plt.plot(iterations_trunc, data_trunc,label=metric)
        plt.legend(loc="lower center",prop={'size':15})
    f.close()


def histeq(im,nbr_bins=256):
    """Histogram equalization"""
    #get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize
    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape)

def imshow_overlay_segmentation(him,img,seg,pred):
    plt.figure(figsize=(20,10))
    plt.subplot(1,3,1)
    plt.title("Image")
    plt.imshow(himg)
    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(img); plt.hold(True)
    plt.imshow(seg, cmap="Blues", alpha=0.3)
    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(img); plt.hold(True)
    plt.imshow(pred, cmap="Reds", alpha=0.3)
    plt.show()

import caffe
print caffe.__file__
# if setup.CAFE_MODE is 'GPU':
    caffe.set_mode_gpu()
# elif setup.CAFE_MODE is 'CPU':
#     caffe.set_mode_cpu()
# else:
#     raise NameError('Invalid CAFE_MODE')


# Load net
try : del solver 
except: pass
solver = caffe.SGDSolver("solver_unet.prototxt")
blobs = solver.net.blobs
params = solver.net.params

import setup

testblobs = solver.test_nets[0].blobs
testparams= solver.test_nets[0].params

if setup.RUN_MODE is 'retrain':
    solver.net.copy_from(setup.PRE_TRAIN_WEIGHTS)
    solver.test_nets[0].copy_from(setup.PRE_TRAIN_WEIGHTS)
    print 'RETRAIN from beginning of weights: ', setup.PRE_TRAIN_WEIGHTS
elif setup.RUN_MODE is 'resume':
    solver.restore(setup.STATE_FILE)
    print 'RESTORE from specific state: ', setup.STATE_FILE
elif setup.RUN_MODE is 'resume-last':
    onlyfiles = next(os.walk(setup.STATE_FOLDER))[2] #dir is your directory path as string
    totalfile = len(onlyfiles) / 2
    if totalfile > 0:
        iterationNum = 500*totalfile
        setup.STATE_FILE = setup.STATE_FOLDER,'_iter_%d.solverstate'%iterationNum
        solver.restore(setup.STATE_FILE)
        print 'RESTORE LAST STATE ', iterationNum
    else:
        raise NameError('Not found any state in %s'%setup.STATE_FOLDER)
else:
    raise NameError('Invalid RUN_MODE')

solver.net.forward()
print 'dice 1', dice(blobs['label'].data[0,0], np.argmax(blobs['score'].data[0],axis=0), label_of_interest=1)
print 'dice 2', dice(blobs['label'].data[0,0], np.argmax(blobs['score'].data[0],axis=0), label_of_interest=2)

solver.test_nets[0].forward()
print 'dice 1', dice(testblobs['label'].data[0,0], np.argmax(testblobs['score'].data[0],axis=0), label_of_interest=1)
print 'dice 2', dice(testblobs['label'].data[0,0], np.argmax(testblobs['score'].data[0],axis=0), label_of_interest=2)

dices = [] #dices for label=1
dices_2 = [] #dices for label=2
losses= []
accuracies=[]
iterations=[]
test_dices=[]
test_dices_2=[]
test_accuracies=[]
i = 0

# TRAIN

def smooth_last_n(arr, n=5, ignore=None):
    """Replaces the last n elements in arr (list) with their average."""
    subarr = np.array(arr[-n:])
    if ignore != None:
        subarr = subarr[subarr != ignore] 
    mean = np.mean(subarr)
    return arr[:-n]+[mean]

iteration_times = []
while True:
    i += 1
    start_ts = time.time()
    solver.step(1)
    end_ts   = time.time()
    iteration_times.append(end_ts-start_ts)

    # Get metrics
    img = blobs['data'].data[0,0]
    seg = blobs['label'].data[0,0]
    pred= np.argmax(blobs['score'].data[0],axis=0)
    dice_score = dice(pred,seg,1)
    dice_score_2 = dice(pred,seg,2) if enable_label_2 else 0
    accuracy_score = np.sum(seg==pred)*1.0 / seg.size
    loss = float(solver.net.blobs['loss'].data)
    
    #Save metrics values
    iterations.append(i)
    dices.append(dice_score if dice_score>-1 else 1)
    dices_2.append(dice_score_2 if dice_score_2>-1 else 1)
    accuracies.append(accuracy_score)
    losses.append(loss)
    
    if i % setup.PLOT_INTERVAL == 0:
        display.clear_output(wait=True)
        
        # Print timing stats
        avg_iteration_time = np.mean(iteration_times)
        iteration_times = []
        
        liver_train_dices = []
        for _ in range(setup.PLOT_INTERVAL):
            solver.test_nets[0].forward()
            test_img = testblobs['data'].data[0,0]
            test_seg = testblobs['label'].data[0,0]
            test_pred= np.argmax(testblobs['score'].data[0], axis=0)

            test_dice_score = dice(test_pred, test_seg, 1)
            test_dice_score_2 = dice(test_pred, test_seg, 2) if enable_label_2 else 0
            test_accuracy_score = np.sum(test_seg==test_pred)*1.0 / test_seg.size

            test_dices.append(test_dice_score if test_dice_score > -1 else 1)
            test_dices_2.append(test_dice_score_2 if test_dice_score_2 > -1 else 1)
            test_accuracies.append(test_accuracy_score)
            
        # Smooth
        iterations = smooth_last_n(iterations  ,n=PLOT_INTERVAL)
        losses     = smooth_last_n(losses      ,n=PLOT_INTERVAL)
        dices      = smooth_last_n(dices       ,n=PLOT_INTERVAL)
        dices_2    = smooth_last_n(dices_2     ,n=PLOT_INTERVAL) if enable_label_2 else []
        accuracies = smooth_last_n(accuracies  ,n=PLOT_INTERVAL)
        test_dices = smooth_last_n(test_dices  ,n=PLOT_INTERVAL)
        test_dices_2=smooth_last_n(test_dices_2,n=PLOT_INTERVAL) if enable_label_2 else []
        test_accuracies = smooth_last_n(test_accuracies,n=PLOT_INTERVAL)
        
        # Print last metrics
        print "Average solver.step duration is", avg_iteration_time
        print 'Loss',losses[-1]
        print '#### ACCURACY ####'
        print 'Train Accuracy', accuracies[-1]
        print 'Test Accuracy', test_accuracies[-1]
        print "#### DICE ####"
        print 'Train dice (label=1)',dices[-1]
        print 'Test dice (label=1)', test_dices[-1]
        if enable_label_2:
            print 'Train dice (label=2)',dices_2[-1]
            print 'Test dice (label=2)', test_dices_2[-1]
        print '\n'

        print "Iteration:", i
        print 'Train accuracy on last image :', np.sum(pred==seg)*1.0/pred.size
        print 'Train dice Label=1 on last image : ', dice(pred,seg,1)
        if enable_label_2:
            print 'Train dice Label=2 on last image : ', dice(pred,seg,2)
        print 'Test accuracy on last image :', np.sum(test_pred==test_seg)*1.0/test_pred.size
        print 'Test dice Label=1 on last image : ', dice(test_pred,test_seg,1)
        if enable_label_2:
            print 'Test dice Label=2 on last image : ', dice(test_pred,test_seg,2)

        pickle.dump(i, open(setup.MONITOR_FOLDER%"i.int",'w'))
        pickle.dump(dices, open(setup.MONITOR_FOLDER%"dices.list",'w'))
        pickle.dump(dices_2, open(setup.MONITOR_FOLDER%"dices_2.list",'w'))
        pickle.dump(losses, open(setup.MONITOR_FOLDER%"losses.list",'w'))
        pickle.dump(accuracies, open(setup.MONITOR_FOLDER%"accuracies.list",'w'))
        pickle.dump(iterations, open(setup.MONITOR_FOLDER%"iterations.list",'w'))
        pickle.dump(test_dices, open(setup.MONITOR_FOLDER%"test_dices.list",'w'))
        pickle.dump(test_dices_2, open(setup.MONITOR_FOLDER%"test_dices_2.list",'w'))
        pickle.dump(test_accuracies, open(setup.MONITOR_FOLDER%"test_accuracies.list",'w'))
    
    