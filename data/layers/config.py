import logging

# Logging level
log_level = logging.WARNING

# Takes only the first n volumes. Useful to create small datasets fast
max_volumes = -1


# Pre-write processing
# Processors applied to images/segmentations right before persisting them to database (after augmentation...etc)
# A processor takes 2 images img and seg, and returns a tuple (img,seg)
# Available processors:
#  - processors.zoomliver_UNET_processor
#  - processors.plain_UNET_processor
#  - processors.histeq_processor
#  - processors.liveronly_label_processor
from numpy_data_layer import processors
# processors_list = [processors.plain_UNET_processor]

# Step 1
# processors_list = [processors.histeq_processor, processors.plain_UNET_processor, processors.liveronly_label_processor]
processors_list = [processors.plain_UNET_processor, processors.liveronly_label_processor]
# Step 2
# processors_list = [processors.histeq_processor, processors.remove_non_liver, processors.zoomliver_UNET_processor]
# processors_list = [processors.remove_non_liver, processors.zoomliver_UNET_processor]

# Shuffle slices and their augmentations globally across the database
# You might want to set to False if dataset = test_set
shuffle_slices = True

# Augmentation factor 
augmentation_factor = 10

# ** Labels order : tissue=0, liver=1, lesion=2
# ** We call a slice "lesion slice" if the MAX label it has is 2
# slice options: liver-lesion, stat-batch, dyn-batch
#
# liver-only:   Include only slices which are labeld with liver or lower (1 or 0)
# lesion-only:  Include only slices which are labeled with lesion or lower (2, 1 or 0)
# liver-lesion: Include only slices which are labeled with liver or lesion (slices with max=2 or with max=1)
# select_slices = "all" # step 1
# select_slices = 'liver-lesion' # step 2
select_slices = 'lesion-only' # for testing step 2, also used for training lesion as liver for step 1

more_small_livers = False
# Percentage of the image, such that any liver small than that is considered small
small_liver_percent = 2

decrease_empty_slices = 0.9

IRCA_NUMPY_BASE_PATH = '/mnt/data/student/3Dircadb1/niftis_segmented_all/'
irca_numpy_all = [\
(301, IRCA_NUMPY_BASE_PATH + "image01.npy", IRCA_NUMPY_BASE_PATH + "label01.npy"),
(302, IRCA_NUMPY_BASE_PATH + "image02.npy", IRCA_NUMPY_BASE_PATH + "label02.npy"),
(303, IRCA_NUMPY_BASE_PATH + "image03.npy", IRCA_NUMPY_BASE_PATH + "label03.npy"),
(304, IRCA_NUMPY_BASE_PATH + "image04.npy", IRCA_NUMPY_BASE_PATH + "label04.npy"),
# (305, IRCA_NUMPY_BASE_PATH + "image05.npy", IRCA_NUMPY_BASE_PATH + "label05.npy"),
(306, IRCA_NUMPY_BASE_PATH + "image06.npy" , IRCA_NUMPY_BASE_PATH + "label06.npy"),
# (307, IRCA_NUMPY_BASE_PATH+"image07.npy",IRCA_NUMPY_BASE_PATH+"label07.npy"),
(308, IRCA_NUMPY_BASE_PATH+"image08.npy",IRCA_NUMPY_BASE_PATH+"label08.npy"),
(309, IRCA_NUMPY_BASE_PATH+"image09.npy",IRCA_NUMPY_BASE_PATH+"label09.npy"),
(310, IRCA_NUMPY_BASE_PATH+"image10.npy",IRCA_NUMPY_BASE_PATH+"label10.npy"),
# (311, IRCA_NUMPY_BASE_PATH+"image11.npy",IRCA_NUMPY_BASE_PATH+"label11.npy"),
(312, IRCA_NUMPY_BASE_PATH+"image12.npy",IRCA_NUMPY_BASE_PATH+"label12.npy"),
(313, IRCA_NUMPY_BASE_PATH+"image13.npy",IRCA_NUMPY_BASE_PATH+"label13.npy"),
# (314, IRCA_NUMPY_BASE_PATH+"image14.npy",IRCA_NUMPY_BASE_PATH+"label14.npy"),
(315, IRCA_NUMPY_BASE_PATH+"image15.npy",IRCA_NUMPY_BASE_PATH+"label15.npy"),
(316, IRCA_NUMPY_BASE_PATH+"image16.npy",IRCA_NUMPY_BASE_PATH+"label16.npy"),
(317, IRCA_NUMPY_BASE_PATH+"image17.npy",IRCA_NUMPY_BASE_PATH+"label17.npy"),
(318, IRCA_NUMPY_BASE_PATH+"image18.npy",IRCA_NUMPY_BASE_PATH+"label18.npy"),
(319, IRCA_NUMPY_BASE_PATH+"image19.npy",IRCA_NUMPY_BASE_PATH+"label19.npy")]
# (320, IRCA_NUMPY_BASE_PATH+"image20.npy",IRCA_NUMPY_BASE_PATH+"label20.npy")]


# Select network datasets
train_dataset = irca_numpy_all[:10]
test_dataset = irca_numpy_all[10:]

