import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 

import sys, os, glob
from multiprocessing import Pool, Process, Queue
import numpy as np
import math
import pdb
from rmath import topi, rotu, rotx, roty, rotz, transu
import IPython
import traceback
from time import sleep

inputdir = "/mnt/data/student/3Dircadb1/niftis_segmented_all_inter"
outputdir = "/mnt/data/student/3Dircadb1/augurment_lesion"

if not os.path.exists(outputdir):
    os.makedirs(outputdir)

def lesionCenter(num):
    if num is 1:
        return zip([(100, 200, 100), (100, 100, 180), (120, 170, 180)], [1, 2, 3])
    elif num is 2:
        return zip([(135, 240, 100)], [1])
    elif num is 3:
        return zip([(120, 220, 175)], [1])
    elif num is 4:
        return zip([(70, 125, 120), (100, 150, 100), (115, 175, 110), (110, 200, 100), (165, 240, 130), (120, 160, 80)]\
                    , [1, 2, 3, 4, 5, 6])
    elif num is 5:
        return zip([(0, 0, 0)], [0])
    elif num is 6:
        return zip([(110, 220, 160)], [1])
    elif num is 7:
        return zip([(0, 0, 0)], [0])
    elif num is 8:
        return zip([(120, 160, 100), (190, 160, 175), (170, 220, 100)], [1, 2, 3])
    elif num is 9:
        return zip([(95, 275, 60), (150, 300, 175)], [1, 2])
    elif num is 10:
        return zip([(100, 275, 85), (140, 220, 110), (210, 240, 170), (170, 210, 175)], [1, 2, 3, 4])
    elif num is 11:
        return zip([(0, 0, 0)], [0])
    elif num is 12:
        return zip([(150, 225, 50)], [1])
    elif num is 13:
        return zip([(100, 200, 140)], [1])
    elif num is 14:
        return zip([(0, 0, 0)], [0])
    elif num is 15:
        return zip([(225, 260, 170)], [1])
    elif num is 16:
        return zip([(100, 190, 95)], [1])
    elif num is 17:
        return zip([(100, 250, 150), (200, 250, 150)], [1, 2])
    elif num is 18:
        return zip([(100, 240, 95)], [1])
    elif num is 19:
        return zip([(0, 0, 0)], [0])
    else:
        return zip([(0, 0, 0)], [0])

def miccaiimshow(img,seg,preds,fname,titles=None, plot_separate_img=True):
	"""Takes raw image img, seg in range 0-2, list of predictions in range 0-2"""
	plt.figure(figsize=(10,5))
	ALPHA=1
	n_plots = len(preds)
	subplot_offset = 0

	plt.set_cmap('gray')

	if plot_separate_img:
		n_plots += 1
		subplot_offset = 1
		plt.subplot(1,n_plots,1)
		# plt.subplots_adjust(wspace=0, hspace=0)
		plt.title("Image")
		plt.axis('off')
		plt.imshow(img,cmap="gray")

	if type(preds) != list:
		preds = [preds]

	for i,pred in enumerate(preds):
		# Order of overaly
		########## OLD
		#lesion= pred==2
		#difflesion = set_minus(seg==2,lesion)
		#liver = set_minus(pred==1, [lesion, difflesion])
		#diffliver = set_minus(seg==1, [liver,lesion,difflesion])
		##########

		lesion= pred==2
		difflesion = np.logical_xor(seg==2,lesion)
		liver = pred==1
		diffliver = np.logical_xor(seg==1, liver)

		plt.subplot(1,n_plots,i+1+subplot_offset)
		title = titles[i] if titles is not None and i < len(titles) else ""
		plt.title(title)
		plt.axis('off')
		plt.imshow(img);plt.hold(True)

		# Liver prediction
		plt.imshow(np.ma.masked_where(liver==0,liver), cmap="Greens",vmin=0.1,vmax=1.2, alpha=ALPHA);plt.hold(True)
		# Liver : Pixels in ground truth, not in prediction
		plt.imshow(np.ma.masked_where(diffliver==0,diffliver), cmap="Spectral",vmin=0.1,vmax=2.2, alpha=ALPHA);plt.hold(True)

		# Lesion prediction
		plt.imshow(np.ma.masked_where(lesion==0,lesion), cmap="Blues",vmin=0.1,vmax=1.2, alpha=ALPHA);plt.hold(True)
		# Lesion : Pixels in ground truth, not in prediction
		plt.imshow(np.ma.masked_where(difflesion==0,difflesion), cmap="Reds",vmin=0.1,vmax=1.5, alpha=ALPHA)

	plt.savefig(fname)
	plt.close()

def buildMatrix(u, theta, sx, sy, sz):
    theta = topi(theta)
    tozero = transu((-sx, -sy, -sz))
    toorigin = transu((sx, sy, sz))
    rotaxis = rotu(u, theta)
    return np.dot(np.dot(toorigin, rotaxis), tozero)

def processTask(fr, to, azip, uzip):

    for i in xrange(fr, to):
        for an, ai in azip:
            for uu, ui in uzip:
                

                imgpath = "%s/image%02d.npy"%(inputdir,i)
                maspath = "%s/label%02d.npy"%(inputdir,i)

                vol = np.load(imgpath)
                vol = vol - np.min(vol) # shift to [0 2047]
                mas = np.load(maspath)
                mas = mas.astype(np.int16)

                w, h, s = vol.shape
                assert w == h
                WIDTH = w
                HEIGHT = h
                SLICE = s
                if SLICE < w:
                    SLICE = w
                    
                padMat = np.zeros([WIDTH, HEIGHT, SLICE]).astype(np.int16)
                padMas = np.zeros([WIDTH, HEIGHT, SLICE]).astype(np.int16)

                if s <= w:
                    SLICE = w
                    beginCenter = (WIDTH - s)
                    minus = beginCenter%2
                    beginCenter = int(beginCenter/2)
                    padMat[:,:,(beginCenter):(SLICE-beginCenter-minus)] = vol
                    padMas[:,:,(beginCenter):(SLICE-beginCenter-minus)] = mas
                else:
                    SLICE = s
                    padMat[:,:,:] = vol
                    padMas[:,:,:] = mas
                del vol
                del mas

                shiftIndex = lesionCenter(i)
                for shift, index in shiftIndex:
                    print "Begin ", i, " pack: ", index, " shift: ", shift, " angle: ", an, " uu: ", uu

                    sx, sy, sz = (0, 0, 0)
                    if index is 0:
                        sx, sy, sz = (WIDTH/2, HEIGHT/2, SLICE/2)
                    else:
                        sx, sy, sz = shift
                        sz = sz + (WIDTH - s)

                    oimgpath = "%s/image%02d_%02d_%02d_%02d.npy"%(outputdir, i, index, ai, ui)
                    omaspath = "%s/label%02d_%02d_%02d_%02d.npy"%(outputdir, i, index, ai, ui)

                    transMat = buildMatrix(uu, an, sx, sy, sz)
                    transMatt = np.linalg.inv(transMat)

                    tarMat = np.zeros([WIDTH, HEIGHT, SLICE]).astype(np.int16)
                    tarMas = np.zeros([WIDTH, HEIGHT, SLICE]).astype(np.int16)

                    for z in xrange(0, SLICE):
                        for y in xrange(0, HEIGHT):
                            for x in xrange(0, WIDTH):
                                vec = np.array([x, y, z, 1]).astype(np.float).reshape([4,1])
                                tar = np.dot(transMatt, vec)
                                tx, ty, tz, p = tar.astype(np.int16)
                                if (tx >= 0 and tx < WIDTH) and (ty >= 0 and ty < HEIGHT) and (tz >= 0 and tz < SLICE):
                                    tarMat[y, x, z] = padMat[ty, tx, tz]
                                    tarMas[y, x, z] = padMas[ty, tx, tz]
                    
                    print "Done Trans ", i

                    # Norm
                    minn = np.min(tarMat)
                    tarMat = tarMat - minn
                    maxx = np.max(tarMat)
                    tarMat = tarMat.astype(np.float64)
                    tarMat = tarMat/maxx
                    tarMat = tarMat*2047.0
                    tarMat = tarMat.astype(np.uint16)

                    np.save(oimgpath, tarMat)
                    np.save(omaspath, tarMas)

                    oexpath = "%s/example/%02d_%02d_%02d_%02d/"%(outputdir, index, i, ai, ui)
                    if not os.path.exists(oexpath):
                        os.makedirs(oexpath)

                    for z in xrange(0, SLICE):
                        fname = "%s/img%03d.png"%(oexpath,z)
                        miccaiimshow(tarMat[:,:,z], tarMas[:,:,z], [tarMas[:,:,z]], fname=fname,titles=["Ground Truth","Prediction"], plot_separate_img=True)
                    
                    print "Done Save ", i

for i in range(1, 20, 2):
    p = Process(target = processTask, args=(i, i+2\
    , zip([15, 30, 45, 60, 75, 90],[1, 2, 3, 4, 5, 6])\
    , zip([(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)\
    , (0.6, 1, 1), (0.3, 1, 1), (1, 0.6, 1), (1, 0.3, 1)\
    , (1, 1, 0.6), (1, 1, 0.3)]\
    , [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
    p.start()
