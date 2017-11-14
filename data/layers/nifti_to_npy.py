#! /usr/bin/env python
'''
Created on Apr 8, 2016

@author: Mohamed.Ezz

This script converts from Nifti (.nii) to Numpy (.npy) files. 
The input to the script is a list of niftis
files are written.

The reason we write nifti to npy is to be sure the npy can be read from disk as a numpy memmap in the NpyDataLayer (caffe data layer)
Using a memmap means we don't have to load the entire array into memory, but only choose individual slices to load and process. 
'''

import sys, os, glob
import numpy as np
import nibabel

###########################
##### 3DIRCA DATASET ######
###########################

IRCA_NUMPY_BASE_PATH = '/Users/Ryan/Data/niftis_segmented_all/'
irca_numpy_all = [\
(301,IRCA_NUMPY_BASE_PATH+"image01.nii",IRCA_NUMPY_BASE_PATH+"label01.nii"),
(302,IRCA_NUMPY_BASE_PATH+"image02.nii",IRCA_NUMPY_BASE_PATH+"label02.nii"),
(303,IRCA_NUMPY_BASE_PATH+"image03.nii",IRCA_NUMPY_BASE_PATH+"label03.nii"),
(304,IRCA_NUMPY_BASE_PATH+"image04.nii",IRCA_NUMPY_BASE_PATH+"label04.nii"),
(305,IRCA_NUMPY_BASE_PATH+"image05.nii",IRCA_NUMPY_BASE_PATH+"label05.nii"),
(306,IRCA_NUMPY_BASE_PATH+"image06.nii",IRCA_NUMPY_BASE_PATH+"label06.nii"),
(307,IRCA_NUMPY_BASE_PATH+"image07.nii",IRCA_NUMPY_BASE_PATH+"label07.nii"),
(308,IRCA_NUMPY_BASE_PATH+"image08.nii",IRCA_NUMPY_BASE_PATH+"label08.nii"),
(309,IRCA_NUMPY_BASE_PATH+"image09.nii",IRCA_NUMPY_BASE_PATH+"label09.nii"),
(310,IRCA_NUMPY_BASE_PATH+"image10.nii",IRCA_NUMPY_BASE_PATH+"label10.nii"),
(311,IRCA_NUMPY_BASE_PATH+"image11.nii",IRCA_NUMPY_BASE_PATH+"label11.nii"),
(312,IRCA_NUMPY_BASE_PATH+"image12.nii",IRCA_NUMPY_BASE_PATH+"label12.nii"),
(313,IRCA_NUMPY_BASE_PATH+"image13.nii",IRCA_NUMPY_BASE_PATH+"label13.nii"),
(314,IRCA_NUMPY_BASE_PATH+"image14.nii",IRCA_NUMPY_BASE_PATH+"label14.nii"),
(315,IRCA_NUMPY_BASE_PATH+"image15.nii",IRCA_NUMPY_BASE_PATH+"label15.nii"),
(316,IRCA_NUMPY_BASE_PATH+"image16.nii",IRCA_NUMPY_BASE_PATH+"label16.nii"),
(317,IRCA_NUMPY_BASE_PATH+"image17.nii",IRCA_NUMPY_BASE_PATH+"label17.nii"),
(318,IRCA_NUMPY_BASE_PATH+"image18.nii",IRCA_NUMPY_BASE_PATH+"label18.nii"),
(319,IRCA_NUMPY_BASE_PATH+"image19.nii",IRCA_NUMPY_BASE_PATH+"label19.nii"),
(320,IRCA_NUMPY_BASE_PATH+"image20.nii",IRCA_NUMPY_BASE_PATH+"label20.nii")]

if __name__ == '__main__':	

	nifti_paths = []
	for volid, volpath, segpath in irca_numpy_all:
		nifti_paths.append(volpath)
		nifti_paths.append(segpath)
		
	for filepath in nifti_paths:
		nifti = nibabel.load(filepath)
		
		out_filepath = filepath.replace(".nii",".npy")
		
		print "Converting", filepath, "to", out_filepath
		array = np.array(nifti.get_data())
		np.save(out_filepath, array)
		
		
		
		