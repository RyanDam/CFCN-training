
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


# In[7]:


import caffe
print caffe.__file__
caffe.set_mode_gpu()


# In[10]:`


# Load net
try : del solver 
except: pass
solver = caffe.SGDSolver("solver_unet.prototxt")
blobs = solver.net.blobs
params = solver.net.params

testblobs = solver.test_nets[0].blobs
testparams= solver.test_nets[0].params

# In[11]:
onlyfiles = next(os.walk('snapshot/'))[2] #dir is your directory path as string
totalfile = len(onlyfiles) / 2
if totalfile > 0:
    # Restore lated state
    iterationNum = 700*totalfile
    STATE_FILE = 'snapshot/_iter_%d.solverstate'%iterationNum
    solver.restore(STATE_FILE)
    print 'RESTORE LAST STATE ',iterationNum
    print STATE_FILE
else:
    # Reload pretrain U-NET model
    WEIGHTS_FILE= 'phseg_v5.caffemodel'
    solver.net.copy_from(WEIGHTS_FILE)
    solver.test_nets[0].copy_from(WEIGHTS_FILE)
    print 'Retrain from beginning'

solver.net.forward()
print 'dice 1', dice(blobs['label'].data[0,0], np.argmax(blobs['score'].data[0],axis=0), label_of_interest=1)
print 'dice 2', dice(blobs['label'].data[0,0], np.argmax(blobs['score'].data[0],axis=0), label_of_interest=2)
# imshow(blobs['data'].data[0,0], blobs['label'].data[0,0], np.argmax(blobs['score'].data[0],axis=0), axis=False,title=["Slice","Ground truth","Prediction"])


# In[ ]:


solver.test_nets[0].forward()
print 'dice 1', dice(testblobs['label'].data[0,0], np.argmax(testblobs['score'].data[0],axis=0), label_of_interest=1)
print 'dice 2', dice(testblobs['label'].data[0,0], np.argmax(testblobs['score'].data[0],axis=0), label_of_interest=2)
# imshow(testblobs['data'].data[0,0,92:-92,92:-92], testblobs['label'].data[0,0], np.argmax(testblobs['score'].data[0],axis=0), axis=False,title=["Slice","Ground truth","Prediction"])


# Config and Initialization
enable_label_2 = False #Set to true when segmenting both liver and lesion (labels=0,1,2)
use_label1_redblue = False # use redblue dice plot. Useful when training cascade Step2 
LOAD_ARRAYS = False # Load arrays from pickled files

if use_label1_redblue:
    label1_color_train, label1_color_test = "blue", "red" 
else:
    label1_color_train, label1_color_test = "#ADB317", "#1C7A34"


PLOT_INTERVAL = 100 # Plot one data point every n iterations
dices = [] #dices for label=1
dices_2 = [] #dices for label=2
losses= []
accuracies=[]
iterations=[]
test_dices=[]
test_dices_2=[]
test_accuracies=[]
i = 0
if LOAD_ARRAYS:
    i=                pickle.load(open("monitor/i.int",'r'))
    dices=            pickle.load(open("monitor/dices.list",'r'))
    if enable_label_2:
        dices_2=          pickle.load(open("monitor/dices_2.list",'r'))
        test_dices_2 =    pickle.load(open("monitor/test_dices_2.list",'r'))
    losses=           pickle.load(open("monitor/losses.list",'r'))
    accuracies=       pickle.load(open("monitor/accuracies.list",'r'))
    iterations =      pickle.load(open("monitor/iterations.list",'r'))
    test_dices =      pickle.load(open("monitor/test_dices.list",'r'))
    test_accuracies = pickle.load(open("monitor/test_accuracies.list",'r'))


# ### To resume run this ###

# In[ ]:


print len(iterations),len(dices),len(dices_2),len(losses),len(accuracies),len(iterations),len(test_dices),len(test_dices_2),len(test_accuracies)
if not enable_label_2:
    test_dices_2 = test_dices
    dices_2 = dices
min_len=min(len(iterations),len(dices),len(dices_2),len(losses),len(accuracies),len(iterations),len(test_dices),len(test_dices_2),len(test_accuracies))
print "Min len",min_len
print 'i',i


# ### then this ###

# In[ ]:


# In case of resumed training, make sure all lists have equal size. Since kernel interruption might cause them to be 
# not equal
#n_ignored_entries = min_len%PLOT_INTERVAL
#min_len -= n_ignored_entries
if len(dices) % 100 != 0 and len(dices) > len(test_dices):
    n_ignored_entries = len(dices) - len(test_dices)
    min_len = len(dices) - n_ignored_entries
    dices = dices[:min_len]
    dices_2=dices_2[:min_len]
    losses= losses[:min_len]
    accuracies=accuracies[:min_len]
    iterations=iterations[:min_len]
    test_dices=test_dices[:min_len]
    test_dices_2=test_dices_2[:min_len]
    test_accuracies=test_accuracies[:min_len]
    i = len(dices) * PLOT_INTERVAL

print len(iterations),len(dices),len(dices_2),len(losses),len(accuracies),len(iterations),len(test_dices),len(test_accuracies)
print 'i',i


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
    
    if i % PLOT_INTERVAL == 0:
        display.clear_output(wait=True)
        
        # Print timing stats
        avg_iteration_time = np.mean(iteration_times)
        iteration_times = []
        
        liver_train_dices = []
        for _ in range(PLOT_INTERVAL):
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
        
        # # Plot
        # fig, ax1=plt.subplots()
        # ax2=ax1.twinx()
        # ax1.set_xlabel("Iterations")
        # ax2.set_ylabel("Dice")
        # ax1.set_ylabel("Loss")
        # ax2.plot(iterations, dices, label="Train Dice - Label=1", color=label1_color_train); plt.hold(True) #dark yellow
        # ax2.plot(iterations, test_dices, label="Test Dice - Label=1", color=label1_color_test); plt.hold(True) #green
        # ax1.plot(iterations, losses, label="Loss", color="black"); plt.hold(True)
        # leg1 = ax2.legend(loc="upper left", bbox_to_anchor=(1.15,1))
        # leg2 = ax1.legend(loc="upper left", bbox_to_anchor=(1.15,0.5))
        # # Make legend clearer
        # for leghandle in leg1.legendHandles+leg2.legendHandles: leghandle.set_linewidth(10.0)
        # plt.show()
        
        
        # if enable_label_2:
        #     fig, ax1=plt.subplots()
        #     ax2=ax1.twinx()
        #     ax1.set_xlabel("Iterations")
        #     ax2.set_ylabel("Dice")
        #     ax1.set_ylabel("Loss")
        #     ax2.plot(iterations, dices_2, label="Train Dice - Label=2",color="blue"); plt.hold(True) #purple #BC23C4
        #     ax2.plot(iterations, test_dices_2, label="Test Dice - Label=2",color="red"); plt.hold(True) #red
        #     ax1.plot(iterations, losses, label="Loss", color="black"); plt.hold(True)
        #     leg1 = ax2.legend(loc="upper left", bbox_to_anchor=(1.15,1))
        #     leg2 = ax1.legend(loc="upper left", bbox_to_anchor=(1.15,0.5))
        #     # Make legend clearer
        #     for leghandle in leg1.legendHandles+leg2.legendHandles: leghandle.set_linewidth(10.0)
        #     plt.show()

        print "Iteration:", i
        print 'Train accuracy on last image :', np.sum(pred==seg)*1.0/pred.size
        print 'Train dice Label=1 on last image : ', dice(pred,seg,1)
        if enable_label_2:
            print 'Train dice Label=2 on last image : ', dice(pred,seg,2)
        # imshow(img, seg, pred, title=["Train Image", "Ground truth", "Prediction"])
        print 'Test accuracy on last image :', np.sum(test_pred==test_seg)*1.0/test_pred.size
        print 'Test dice Label=1 on last image : ', dice(test_pred,test_seg,1)
        if enable_label_2:
            print 'Test dice Label=2 on last image : ', dice(test_pred,test_seg,2)
        # imshow(test_img, test_seg, test_pred, title=["Test Image", "Ground truth", "Prediction"])

        pickle.dump(i, open("monitor/i.int",'w'))
        pickle.dump(dices, open("monitor/dices.list",'w'))
        pickle.dump(dices_2, open("monitor/dices_2.list",'w'))
        pickle.dump(losses, open("monitor/losses.list",'w'))
        pickle.dump(accuracies, open("monitor/accuracies.list",'w'))
        pickle.dump(iterations, open("monitor/iterations.list",'w'))
        pickle.dump(test_dices, open("monitor/test_dices.list",'w'))
        pickle.dump(test_dices_2, open("monitor/test_dices_2.list",'w'))
        pickle.dump(test_accuracies, open("monitor/test_accuracies.list",'w'))
    
    

# ##########################################################################################
# # ## ---- End of training notebook (the rest is one-off analysis) ##
# ##########################################################################################

# # In[ ]:


# td2pure = td2[np.logical_and(td2!=1.0, td2!=0.0)]


# # In[ ]:


# np.count_nonzero(test_dices_2==1.0)


# # In[ ]:


# fidx=1
# layer_name='u0c'
# for fidx in range(blobs[layer_name].data.shape[1]):
#     last_layer_img = blobs[layer_name].data[0,fidx, :,:]
#     imshow(last_layer_img)


# # In[ ]:


# for i in iterations:
#     print i


# # In[ ]:


# lyr=solver.net.layers[64]
# lyr.type
# lyr.blobs.__len__()


# # In[ ]:


# for param_name in solver.net.params:
#     print param_name,"\t",solver.net.params[param_name][0].data.shape


# # In[ ]:


# show_kernels(solver.net.params["conv_d0a-b"][0].data)


# # In[ ]:


# for idx in range(blobs['d2c'].data.shape[1]):
#     imshow(blobs['d2c'].data[0,idx])


# # In[ ]:


# show_kernels(blobs['d0b'].data)


# # In[ ]:


# blobs


# # In[ ]:


# print solver.net.params["conv_d0a-b"][0].data.shape
# imshow(solver.net.params["conv_d0a-b"][0].data[7,0])


# # In[ ]:


# imshow(blobs['data'].data[0,0])


# # In[ ]:


# from scipy.signal import convolve2d
# filter_idx = 7
# image = blobs['data'].data[0,0]
# bias = solver.net.params["conv_d0a-b"][1].data[filter_idx]
# kernel = solver.net.params["conv_d0a-b"][0].data[filter_idx,0]
# result = convolve2d(image, kernel) + bias
# print bias
# imshow(result)


# # In[ ]:


# import scipy.signal.convolve2d


# # In[ ]:


# print blobs['d0b'].data.shape
# imshow(blobs['d0b'].data[0,15],blobs['d0b'].data[0,7])


# # ## Get iteration with best test dice ##

# # In[ ]:


# dice_iter = zip(test_dices,iterations)
# dice_iter = sorted(dice_iter, key=lambda t:t[0], reverse=True)
# for ji in range(10):
#     print str(ji+1)+'th best test Dice:\t',round(dice_iter[ji][0],3),'\tAt iteration:\t',dice_iter[ji][1]


# # In[ ]:


# # Save plots
# import pickle
# pickle.dump(i, open("monitor/i.int",'w'))
# pickle.dump(dices, open("monitor/dices.list",'w'))
# pickle.dump(dices_2, open("monitor/dices_2.list",'w'))
# pickle.dump(losses, open("monitor/losses.list",'w'))
# pickle.dump(accuracies, open("monitor/accuracies.list",'w'))
# pickle.dump(iterations, open("monitor/iterations.list",'w'))
# pickle.dump(test_dices, open("monitor/test_dices.list",'w'))
# pickle.dump(test_dices_2, open("monitor/test_dices_2.list",'w'))
# pickle.dump(test_accuracies, open("monitor/test_accuracies.list",'w'))


# # In[ ]:


# mean = protobinary_to_array("mean.protobinary")


# # ## Predicting Training examples ##

# # In[ ]:


# for _ in range(20):
#     solver.net.forward()
#     img = (blobs['data'].data[0,0]+mean)[92:480,92:480]
#     himg =histeq(img)
#     seg = blobs['label'].data[0,0]
#     pred= np.argmax(blobs['score'].data[0],axis=0)

#     imshow_overlay_segmentation(himg,img,seg,pred)


# # 
# # 
# # 
# # 
# # 
# # 
# # # Predict TEST examples #

# # In[ ]:


# for _ in range(20):
#     solver.test_nets[0].forward()
#     img = (testblobs['data'].data[0,0]+mean)[92:480,92:480]
#     himg =histeq(img)
#     seg = testblobs['label'].data[0,0]
#     pred= np.argmax(testblobs['score'].data[0],axis=0)

#     imshow_overlay_segmentation(himg,img,seg,pred)


# # # Avg Dice score over slices #

# # In[ ]:


# #dices_lesions_ = []
# for _ in range(1000):
#     solver.test_nets[0].forward()
#     seg = solver.test_nets[0].blobs['label'].data[0,0]
#     pred = solver.test_nets[0].blobs['score'].data[0].argmax(0)
#     dice_lesion_ = dice(pred,seg,label_of_interest=1)

#     if(dice_lesion_ > -1):
#         dices_lesions_.append(dice_lesion_)
#     print "Average TEST dice lesion: ", np.average(dices_lesions_)

# print "FINAL Average TEST dice lesion: ", np.average(dices_lesions_)


# # # Avg Dice score over individual Lesions #

# # In[ ]:


# import sys
# import scipy.spatial.distance
# import scipy.ndimage
# import scipy.ndimage.measurements
# from collections import defaultdict
# def dice_separate_lesions(seg,pred, plot=False):
#     """Returns Avg dice of lesion structures and weight to assign to this avg dice."""
#     #Ignore liver
#     if np.unique(seg).size > 2:
#         seg[seg==1] = 0
#         seg[seg==2] = 1
#     if np.unique(pred).size > 2:
#         pred[pred==1] = 0
#         pred[pred==2] = 1
#     # First component is always background
#     seg[0,0] = 0
#     pred[0,0] = 0
#     # Get connected components
#     comps_seg, num_comps_seg = scipy.ndimage.label(seg)
#     comps_pred, num_comps_pred = scipy.ndimage.label(pred)
#     #print 'Found n connected components in ground truth (not including bg) :', num_comps_seg
#     if plot: imshow(comps_seg, comps_pred, cmap="Spectral", title=['Components in Ground Truth','Components in Prediction'])
#     # Get component centroids
#     centroids_seg = scipy.ndimage.measurements.center_of_mass(seg, comps_seg, range(1, num_comps_seg+1))
#     centroids_pred = scipy.ndimage.measurements.center_of_mass(pred, comps_pred, range(1, num_comps_pred+1))
#     # round to nearest 2 decimals (otherwise we might have problems removing from list by-value due to fp inaccuracies)
#     centroids_seg = map(lambda t:(round(t[0],2), round(t[1],2)), centroids_seg)
#     centroids_pred = map(lambda t:(round(t[0],2), round(t[1],2)), centroids_pred)
    
#     def plot_centroids(comps_img, centroids, title, w=5):
#         centroid_img = np.ones(comps_img.shape)
#         for x,y in centroids:
#             centroid_img[x-w:x+w, y-w:y+w] = 0
#         plt.title(title)
#         plt.imshow(comps_img, cmap="Spectral"); plt.hold(True)
#         plt.imshow(centroid_img,cmap="Reds",alpha=0.5)
#         plt.show()
    
    
#     if plot: plot_centroids(comps_seg, centroids_seg, "Centroids in Ground Truth")
#     if plot: plot_centroids(comps_pred, centroids_pred, "Centroids in Prediction")
    
#     #### Get Average dice ####
#     def get_closest(xy, list_xy, except_at_idx):
#         """Returns the index of coordinate in list_xy that is closest to xy (euclidean distance)
#         example: get_closest((100,100), [(3,4), (5,9), (101,102), (9999,9999)]) = 2
#         because (101,102) is the closest to (100,100).
#         except_at_idx is a list of coordinate indices to ignore in list_xy"""
#         closest_idx = -1
#         min_dist = sys.maxint
#         for i, xy_dest in enumerate(list_xy):
#             if i in except_at_idx:
#                 continue
#             dist = scipy.spatial.distance.euclidean(xy, xy_dest)
#             if dist < min_dist:
#                 closest_idx = i
#                 min_dist = dist
#         #print xy, map(lambda t:(round(t[0]),round(t[1])),list_xy), closest_idx
#         return closest_idx
    
#     dices = []
#     consumed_lesions_idx = [] #indices of lesions already consumed.
#     # Iterate after bg component
#     for i in range(num_comps_pred):
#         # Add 0 dice to false positives!
#         if len(centroids_seg) == 0:
#             dices.append(0)
#             continue
#         current_xy = centroids_pred[i]
#         closest_component = get_closest(current_xy, centroids_seg, except_at_idx=consumed_lesions_idx)
#         consumed_lesions_idx.append(closest_component)
#         #mask out other components 
#         one_lesion_pred = np.clip(comps_pred == i, 0, 1)
#         one_lesion_seg = np.clip(comps_seg == closest_component, 0, 1)
#         dices.append(dice(one_lesion_pred, one_lesion_seg, label_of_interest = 1))
    
#     # Add 0 dice for false negatives
#     if len(centroids_seg)-len(consumed_lesions_idx) > 0:
#         dices.extend([0]*(len(centroids_seg)-len(consumed_lesions_idx)))

#     return np.mean(dices), len(dices)


# # In[ ]:


# dices_lesions_ = []
# weights = []
# for _ in range(200):
#     solver.test_nets[0].forward()
#     seg = solver.test_nets[0].blobs['label'].data[0,0]
#     pred = solver.test_nets[0].blobs['score'].data[0].argmax(0)
#     dice_lesion_,weight = dice_separate_lesions(seg,pred)

#     if(dice_lesion_ > -1):
#         dices_lesions_.append(dice_lesion_)
#         weights.append(weight)

# total = np.multiply(dices_lesions_, weights)
# print "Average TEST dice lesion: ", np.average(total)


# # ## Changing threshold (instead of 0.5) ##

# # In[ ]:


# solver.test_nets[0].forward()
# img = (testblobs['data'].data[0,0]+mean)[92:480,92:480]
# himg =histeq(img)
# seg = testblobs['label'].data[0,0]
# pred= np.argmax(testblobs['score'].data[0],axis=0)
# imshow_overlay_segmentation(himg,img,seg,pred)


# # In[ ]:


# prob = testblobs['prob'].data[0,1]
# imshow(prob)
# pred_t = prob>0.7
# print dice(seg,pred)
# print dice(seg,pred_t)
# imshow(pred_t,seg)
# #prob[np.logical_and(prob>0.4, prob<0.6)].size*1.0/prob.size


# #imshow_overlay_segmentation(himg,img,seg,pred)


# # In[ ]:


# def softmax(a1,a2):
#     s1 = np.exp(a1)
#     s2 = np.exp(a2)
#     sm= s1+s2
#     return s1/sm , s2/sm

# thresholds = np.linspace(0,1,40) #20 thresholds
# dices_athalf = []
# dices_bythreshold = defaultdict(list) # {0.5:[list of dices], 0.6:[list of dices]}
# for _ in range(2000):
#     solver.net.forward()
#     img = (blobs['data'].data[0,0]+mean)[92:480,92:480]
#     himg =histeq(img)
#     seg = blobs['label'].data[0,0]
#     prob = softmax(blobs['score'].data[0,0],blobs['score'].data[0,1])[1] #probability being a lesion
#     dices_athalf.append(dice(seg,prob>0.5))
#     for t in thresholds :
#         pred_t = prob > t
#         dice_score = dice(seg,pred_t)
#         dices_bythreshold[t].append(dice_score)
        
# # Aggregate dices over slices for each threshold
# avgdices = []
# for t in thresholds:
#     avgdices.append(np.average(dices_bythreshold[t]))

# plt.plot(thresholds,avgdices)
# print 'Found max dice at threshold :', thresholds[np.argmax(avgdices)]
# print 'Found max dice score :', np.max(avgdices)
# print 'Vs. the dice at 0.5 which equals :', np.average(dices_athalf)


# # In[ ]:


# THRESHOLD = 0.820512820513
# dices = []
# for _ in range(2000):
#     solver.test_nets[0].forward()
#     img = (testblobs['data'].data[0,0]+mean)[92:480,92:480]
#     seg = testblobs['label'].data[0,0]
#     prob = testblobs['prob'].data[0,1] #probability being a lesion
#     dice_score = dice(seg,prob> THRESHOLD)
#     dices.append(dice_score)
    
# print 'Average TEST threshold :', np.average(dices)


# # In[ ]:



# dices = []
# for _ in range(2000):
#     solver.test_nets[0].forward()
#     img = (testblobs['data'].data[0,0]+mean)[92:480,92:480]
#     seg = testblobs['label'].data[0,0]
#     pred = np.argmax(testblobs['prob'].data[0], axis=0) #probability being a lesion
#     dice_score = dice(seg,pred)
#     dices.append(dice_score)
    
# print 'Average TEST threshold :', np.average(dices)


# # In[ ]:


# thresholds = np.linspace(0,1,40) #20 thresholds
# dices_athalf = []
# dices_bythreshold = defaultdict(list) # {0.5:[list of dices], 0.6:[list of dices]}
# for _ in range(2000):
#     solver.test_nets[0].forward()
#     img = (testblobs['data'].data[0,0]+mean)[92:480,92:480]
#     himg =histeq(img)
#     seg = testblobs['label'].data[0,0]
#     prob = testblobs['prob'].data[0,1] #probability being a lesion
#     dices_athalf.append(dice(seg,prob>0.5))
#     for t in thresholds :
#         pred_t = prob > t
#         dice_score = dice(seg,pred_t)
#         dices_bythreshold[t].append(dice_score)
        
# # Aggregate dices over slices for each threshold
# avgdices = []
# for t in thresholds:
#     avgdices.append(np.average(dices_bythreshold[t]))

# plt.plot(thresholds,avgdices)
# print 'Found max dice at threshold :', thresholds[np.argmax(avgdices)]
# print 'Found max dice score :', np.max(avgdices)
# print 'Vs. the dice at 0.5 which equals :', np.average(dices_athalf)


# # # Prototxts #

# # In[ ]:


# get_ipython().system(u'cat solver_unet.prototxt')


# # In[ ]:


# get_ipython().system(u'cat unet-overfit.prototxt')

