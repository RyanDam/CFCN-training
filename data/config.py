import logging
import sys
sys.path.append('../saratan/')

# Logging level
log_level = logging.WARNING
# Number of CPUs used for parallel processing

N_PROC = 14

# Path of created database
# This can be a list with multiple paths, but also dataset should be a list of same size
lmdb_path = ["../ID40-UNET-LiverOnlyLabel-572-liverlesion-AugSmallLiver/train", "../ID40-UNET-LiverOnlyLabel-572-liverlesion-AugSmallLiver/validation"]
# Database type : lmdb or leveldb
backend = "lmdb" 
# Takes only the first n volumes. Useful to create small datasets fast
max_volumes = -1

# whether to increase samples whose liver is small (less than 
# If liver pixels is less than small_liver_percent, it will be considered small liver
augment_small_liver = True
small_liver_percent = 2.5 

# Shuffle slices and their augmentations globally across the database
# You might want to set to False if dataset = test_set
shuffle_slices = True

# Augmentation factor 
augmentation_factor = 11

# Image/Seg shape
slice_shape = (388,388)
# Pre-write processing
# Processors applied to images/segmentations right before persisting them to database (after augmentation...etc)
# A processor takes 2 images img and seg, and returns a tuple (img,seg)
# Available processors:
#  - processors.zoomliver_UNET_processor
#  - processors.plain_UNET_processor
#  - processors.histeq_processor
#  - processors.liveronly_label_processor

# - proexessors.filter_preprocessor
import create_ctdata as processors
processors_list = [processors.plain_UNET_processor,processors.filter_preprocessor]
#processors_list = [processors.plain_UNET_processor]

# Hounsfield Unit Windowing
# Apply static or dynamic Windowing to the CT data
#ct_window_type='dyn'
#ct_window_type_min=0.1
#ct_window_type_max=0.3

ct_window_type='stat'
ct_window_type_min=-100
ct_window_type_max=200



# Image Filtering
# Filter the Images as preprocessing

filter_type='bilateral'


# Augmentation factor 
augmentation_factor = 17

# whether to increase samples whose liver is small (less than 
augment_small_liver = True


# ** Labels order : tissue=0, liver=1, lesion=2
# ** We call a slice "lesion slice" if the MAX label it has is 2
# slice options: liver-lesion, stat-batch, dyn-batch
#
# liver-only:   Include only slices which are labeld with liver or lower (1 or 0)
# lesion-only:  Include only slices which are labeled with lesion or lower (2, 1 or 0)
# liver-lesion: Include only slices which are labeled with liver or lesion (slices with max=2 or with max=1)
# all slices: Include slices which are not liver or lesion with a percentage irrelevant_slice_include_prob to enable this feature uncomment
#irrelevant_slice_include_prob=10

select_slices = "liver-lesion"

###########################
##### 3DIRCA DATASET ######
###########################
IRCA_BASE_PATH = '../3Dircadb1/niftis_segmented_lesions/'
irca_all= [\
(301,IRCA_BASE_PATH+"image01.nii",IRCA_BASE_PATH+"label01.nii",[0.57,0.57,1.6]),
(302,IRCA_BASE_PATH+"image02.nii",IRCA_BASE_PATH+"label02.nii",[0.78,0.78,1.6]),
(303,IRCA_BASE_PATH+"image03.nii",IRCA_BASE_PATH+"label03.nii",[0.62,0.62,1.25]),
(304,IRCA_BASE_PATH+"image04.nii",IRCA_BASE_PATH+"label04.nii",[0.74,0.74,2.]),
#(305,IRCA_BASE_PATH+"image05.nii",IRCA_BASE_PATH+"label05.nii",[0.78,0.78,1.6]),
(306,IRCA_BASE_PATH+"image06.nii",IRCA_BASE_PATH+"label06.nii",[0.78,0.78,1.6]),
#(307,IRCA_BASE_PATH+"image07.nii",IRCA_BASE_PATH+"label07.nii",[0.78,0.78,1.6]),
(308,IRCA_BASE_PATH+"image08.nii",IRCA_BASE_PATH+"label08.nii",[0.56,0.56,1.6]),
(309,IRCA_BASE_PATH+"image09.nii",IRCA_BASE_PATH+"label09.nii",[0.87,0.87,2.]),
(310,IRCA_BASE_PATH+"image10.nii",IRCA_BASE_PATH+"label10.nii",[0.73,0.73,1.6]),
#(311,IRCA_BASE_PATH+"image11.nii",IRCA_BASE_PATH+"label11.nii",[0.72,0.72,1.6]),
(312,IRCA_BASE_PATH+"image12.nii",IRCA_BASE_PATH+"label12.nii",[0.68,0.68,1.]),
(313,IRCA_BASE_PATH+"image13.nii",IRCA_BASE_PATH+"label13.nii",[0.67,0.67,1.6]),
#(314,IRCA_BASE_PATH+"image14.nii",IRCA_BASE_PATH+"label14.nii",[0.72,0.72,1.6]),
(315,IRCA_BASE_PATH+"image15.nii",IRCA_BASE_PATH+"label15.nii",[0.78,0.78,1.6]),
(316,IRCA_BASE_PATH+"image16.nii",IRCA_BASE_PATH+"label16.nii",[0.7,0.7,1.6]),
(317,IRCA_BASE_PATH+"image17.nii",IRCA_BASE_PATH+"label17.nii",[0.74,0.74,1.6]),
(318,IRCA_BASE_PATH+"image18.nii",IRCA_BASE_PATH+"label18.nii",[0.74,0.74,2.5]),
(319,IRCA_BASE_PATH+"image19.nii",IRCA_BASE_PATH+"label19.nii",[0.7,0.7,4.])]
#(320,IRCA_BASE_PATH+"image20.nii",IRCA_BASE_PATH+"label20.nii",[0.81,0.81,2.])]

# Select dataset
dataset = dataset_fire3
