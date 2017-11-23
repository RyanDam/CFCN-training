import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt 

import numpy as np
import math
import pdb
from rmath import topi, rotu, rotx, roty, rotz, transu
import IPython
import traceback

inputdir = "/mnt/data/student/3Dircadb1/niftis_segmented_all_inter"
outputdir = "/mnt/data/student/augurment"
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

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

def buildMatrix(u, theta, W, H, S):
    theta = topi(theta)
    tozero = transu((-W/2, -H/2, -S/2))
    toorigin = transu((W/2, H/2, S/2))
    rotaxis = rotu(u, theta)
    return np.dot(np.dot(toorigin, rotaxis), tozero)

try:
    for i in xrange(1, 21):
        for an in [1, 2]:
            for uu, ui in zip([(1, 1, 1), (1, 1, -1)], [0, 1])

            imgpath = "%s/image%02d.npy"%inputdir,i
            maspath = "%s/label%02d.npy"%inputdir,i
            oimgpath = "%s/image%02d.npy"%outputdir,(i + 20*an + 20*ui)
            omaspath = "%s/label%02d.npy"%outputdir,(i + 20*an + 20*ui)

            vol = np.load(imgpath)
            vol = vol - np.min(vol) # shift to [0 2047]
            mas = np.load(maspath)
            mas = mas.astype(np.int16)

            w, h, s = vol.shape
            assert w == h
            WIDTH = w
            HEIGHT = h
            SLICE = w

            padMat = np.zeros([WIDTH, HEIGHT, SLICE]).astype(np.int16)
            padMas = np.zeros([WIDTH, HEIGHT, SLICE]).astype(np.int16)

            beginCenter = (WIDTH - s)
            minus = beginCenter%2
            beginCenter = int(beginCenter/2)

            padMat[:,:,(beginCenter):(SLICE-beginCenter-minus)] = vol
            padMas[:,:,(beginCenter):(SLICE-beginCenter-minus)] = mas
            del vol
            del mas

            transMat = buildMatrix(uu, an*30, WIDTH, HEIGHT, SLICE)
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
                print "Done slice ", z

            oexpath = "%s/example/%02d/"%outputdir,(i + 20*an + 20*ui)
            if not os.path.exists(oexpath):
                os.makedirs(oexpath)

            for z in xrange(0, SLICE):
                
                fname = "%s/img%03d.png"%oexpath,z
                miccaiimshow(tarMat[:,:,z], tarMas[:,:,z], [tarMas[:,:,z]], fname=fname,titles=["Ground Truth","Prediction"], plot_separate_img=True)
                print "Done save ", z
            
            pdb.set_trace()
            print 1

except:
    traceback.print_exc()
    print '\n'
    IPython.embed()