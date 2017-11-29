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
processors_list = [processors.plain_UNET_processor, processors.liveronly_label_processor]
# Step 2
# processors_list = [processors.remove_non_liver, processors.zoomliver_UNET_processor] # original in paper
# processors_list = [processors.remove_non_liver, processors.plain_UNET_processor]

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
# step 1
select_slices = 'lesion-only' # For step 1.1, learn liver with lesion is one
# select_slices = 'liver-lesion' # For step 1.2 learn all healhty liver
# step 2
# select_slices = 'lesion-only'

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
(305, IRCA_NUMPY_BASE_PATH + "image05.npy", IRCA_NUMPY_BASE_PATH + "label05.npy"),
(306, IRCA_NUMPY_BASE_PATH + "image06.npy" , IRCA_NUMPY_BASE_PATH + "label06.npy"),
(307, IRCA_NUMPY_BASE_PATH+"image07.npy",IRCA_NUMPY_BASE_PATH+"label07.npy"),
(308, IRCA_NUMPY_BASE_PATH+"image08.npy",IRCA_NUMPY_BASE_PATH+"label08.npy"),
(309, IRCA_NUMPY_BASE_PATH+"image09.npy",IRCA_NUMPY_BASE_PATH+"label09.npy"),
(310, IRCA_NUMPY_BASE_PATH+"image10.npy",IRCA_NUMPY_BASE_PATH+"label10.npy"),
(311, IRCA_NUMPY_BASE_PATH+"image11.npy",IRCA_NUMPY_BASE_PATH+"label11.npy"),
(312, IRCA_NUMPY_BASE_PATH+"image12.npy",IRCA_NUMPY_BASE_PATH+"label12.npy"),
(313, IRCA_NUMPY_BASE_PATH+"image13.npy",IRCA_NUMPY_BASE_PATH+"label13.npy"),
(314, IRCA_NUMPY_BASE_PATH+"image14.npy",IRCA_NUMPY_BASE_PATH+"label14.npy"),
(315, IRCA_NUMPY_BASE_PATH+"image15.npy",IRCA_NUMPY_BASE_PATH+"label15.npy"),
(316, IRCA_NUMPY_BASE_PATH+"image16.npy",IRCA_NUMPY_BASE_PATH+"label16.npy"),
(317, IRCA_NUMPY_BASE_PATH+"image17.npy",IRCA_NUMPY_BASE_PATH+"label17.npy"),
(318, IRCA_NUMPY_BASE_PATH+"image18.npy",IRCA_NUMPY_BASE_PATH+"label18.npy"),
(319, IRCA_NUMPY_BASE_PATH+"image19.npy",IRCA_NUMPY_BASE_PATH+"label19.npy"),
(320, IRCA_NUMPY_BASE_PATH+"image20.npy",IRCA_NUMPY_BASE_PATH+"label20.npy")]

BASE_AU_PATH = "/mnt/data/student/3Dircadb1/augurment_all"
irca_augumentation_all = [\
(121, BASE_AU_PATH+"image21.npy", BASE_AU_PATH+"label21.npy"),
(122, BASE_AU_PATH+"image22.npy", BASE_AU_PATH+"label22.npy"),
(123, BASE_AU_PATH+"image23.npy", BASE_AU_PATH+"label23.npy"),
(124, BASE_AU_PATH+"image24.npy", BASE_AU_PATH+"label24.npy"),
(125, BASE_AU_PATH+"image25.npy", BASE_AU_PATH+"label25.npy"),
(126, BASE_AU_PATH+"image26.npy", BASE_AU_PATH+"label26.npy"),
(127, BASE_AU_PATH+"image27.npy", BASE_AU_PATH+"label27.npy"),
(128, BASE_AU_PATH+"image28.npy", BASE_AU_PATH+"label28.npy"),
(129, BASE_AU_PATH+"image29.npy", BASE_AU_PATH+"label29.npy"),
(130, BASE_AU_PATH+"image30.npy", BASE_AU_PATH+"label30.npy"),
(131, BASE_AU_PATH+"image31.npy", BASE_AU_PATH+"label31.npy"),
(132, BASE_AU_PATH+"image32.npy", BASE_AU_PATH+"label32.npy"),
(133, BASE_AU_PATH+"image33.npy", BASE_AU_PATH+"label33.npy"),
(134, BASE_AU_PATH+"image34.npy", BASE_AU_PATH+"label34.npy"),
(135, BASE_AU_PATH+"image35.npy", BASE_AU_PATH+"label35.npy"),
(136, BASE_AU_PATH+"image36.npy", BASE_AU_PATH+"label36.npy"),
(137, BASE_AU_PATH+"image37.npy", BASE_AU_PATH+"label37.npy"),
(138, BASE_AU_PATH+"image38.npy", BASE_AU_PATH+"label38.npy"),
(139, BASE_AU_PATH+"image39.npy", BASE_AU_PATH+"label39.npy"),
(140, BASE_AU_PATH+"image40.npy", BASE_AU_PATH+"label40.npy"),
(141, BASE_AU_PATH+"image41.npy", BASE_AU_PATH+"label41.npy"),
(142, BASE_AU_PATH+"image42.npy", BASE_AU_PATH+"label42.npy"),
(143, BASE_AU_PATH+"image43.npy", BASE_AU_PATH+"label43.npy"),
(144, BASE_AU_PATH+"image44.npy", BASE_AU_PATH+"label44.npy"),
(145, BASE_AU_PATH+"image45.npy", BASE_AU_PATH+"label45.npy"),
(146, BASE_AU_PATH+"image46.npy", BASE_AU_PATH+"label46.npy"),
(147, BASE_AU_PATH+"image47.npy", BASE_AU_PATH+"label47.npy"),
(148, BASE_AU_PATH+"image48.npy", BASE_AU_PATH+"label48.npy"),
(149, BASE_AU_PATH+"image49.npy", BASE_AU_PATH+"label49.npy"),
(150, BASE_AU_PATH+"image50.npy", BASE_AU_PATH+"label50.npy"),
(151, BASE_AU_PATH+"image51.npy", BASE_AU_PATH+"label51.npy"),
(152, BASE_AU_PATH+"image52.npy", BASE_AU_PATH+"label52.npy"),
(153, BASE_AU_PATH+"image53.npy", BASE_AU_PATH+"label53.npy"),
(154, BASE_AU_PATH+"image54.npy", BASE_AU_PATH+"label54.npy"),
(155, BASE_AU_PATH+"image55.npy", BASE_AU_PATH+"label55.npy"),
(156, BASE_AU_PATH+"image56.npy", BASE_AU_PATH+"label56.npy"),
(157, BASE_AU_PATH+"image57.npy", BASE_AU_PATH+"label57.npy"),
(158, BASE_AU_PATH+"image58.npy", BASE_AU_PATH+"label58.npy"),
(159, BASE_AU_PATH+"image59.npy", BASE_AU_PATH+"label59.npy"),
(160, BASE_AU_PATH+"image60.npy", BASE_AU_PATH+"label60.npy"),
(161, BASE_AU_PATH+"image61.npy", BASE_AU_PATH+"label61.npy"),
(162, BASE_AU_PATH+"image62.npy", BASE_AU_PATH+"label62.npy"),
(163, BASE_AU_PATH+"image63.npy", BASE_AU_PATH+"label63.npy"),
(164, BASE_AU_PATH+"image64.npy", BASE_AU_PATH+"label64.npy"),
(165, BASE_AU_PATH+"image65.npy", BASE_AU_PATH+"label65.npy"),
(166, BASE_AU_PATH+"image66.npy", BASE_AU_PATH+"label66.npy"),
(167, BASE_AU_PATH+"image67.npy", BASE_AU_PATH+"label67.npy"),
(168, BASE_AU_PATH+"image68.npy", BASE_AU_PATH+"label68.npy"),
(169, BASE_AU_PATH+"image69.npy", BASE_AU_PATH+"label69.npy"),
(170, BASE_AU_PATH+"image70.npy", BASE_AU_PATH+"label70.npy"),
(171, BASE_AU_PATH+"image71.npy", BASE_AU_PATH+"label71.npy"),
(172, BASE_AU_PATH+"image72.npy", BASE_AU_PATH+"label72.npy"),
(173, BASE_AU_PATH+"image73.npy", BASE_AU_PATH+"label73.npy"),
(174, BASE_AU_PATH+"image74.npy", BASE_AU_PATH+"label74.npy"),
(175, BASE_AU_PATH+"image75.npy", BASE_AU_PATH+"label75.npy"),
(176, BASE_AU_PATH+"image76.npy", BASE_AU_PATH+"label76.npy"),
(177, BASE_AU_PATH+"image77.npy", BASE_AU_PATH+"label77.npy"),
(178, BASE_AU_PATH+"image78.npy", BASE_AU_PATH+"label78.npy"),
(179, BASE_AU_PATH+"image79.npy", BASE_AU_PATH+"label79.npy"),
(180, BASE_AU_PATH+"image80.npy", BASE_AU_PATH+"label80.npy")]

# # Select network datasets
# train_dataset = [\
# (301, IRCA_NUMPY_BASE_PATH + "image01.npy", IRCA_NUMPY_BASE_PATH + "label01.npy"),
# (302, IRCA_NUMPY_BASE_PATH + "image02.npy", IRCA_NUMPY_BASE_PATH + "label02.npy"),
# (303, IRCA_NUMPY_BASE_PATH + "image03.npy", IRCA_NUMPY_BASE_PATH + "label03.npy"),
# (304, IRCA_NUMPY_BASE_PATH + "image04.npy", IRCA_NUMPY_BASE_PATH + "label04.npy"),
# (305, IRCA_NUMPY_BASE_PATH + "image05.npy", IRCA_NUMPY_BASE_PATH + "label05.npy"),
# (306, IRCA_NUMPY_BASE_PATH + "image06.npy" , IRCA_NUMPY_BASE_PATH + "label06.npy"),
# (307, IRCA_NUMPY_BASE_PATH+"image07.npy",IRCA_NUMPY_BASE_PATH+"label07.npy"),
# (308, IRCA_NUMPY_BASE_PATH+"image08.npy",IRCA_NUMPY_BASE_PATH+"label08.npy"),
# (309, IRCA_NUMPY_BASE_PATH+"image09.npy",IRCA_NUMPY_BASE_PATH+"label09.npy"),
# (310, IRCA_NUMPY_BASE_PATH+"image10.npy",IRCA_NUMPY_BASE_PATH+"label10.npy"),
# (311, IRCA_NUMPY_BASE_PATH+"image11.npy",IRCA_NUMPY_BASE_PATH+"label11.npy"),
# (312, IRCA_NUMPY_BASE_PATH+"image12.npy",IRCA_NUMPY_BASE_PATH+"label12.npy"),
# (313, IRCA_NUMPY_BASE_PATH+"image13.npy",IRCA_NUMPY_BASE_PATH+"label13.npy"),
# (314, IRCA_NUMPY_BASE_PATH+"image14.npy",IRCA_NUMPY_BASE_PATH+"label14.npy"),
# (121, BASE_AU_PATH+"image21.npy", BASE_AU_PATH+"label21.npy"),
# (122, BASE_AU_PATH+"image22.npy", BASE_AU_PATH+"label22.npy"),
# (123, BASE_AU_PATH+"image23.npy", BASE_AU_PATH+"label23.npy"),
# (124, BASE_AU_PATH+"image24.npy", BASE_AU_PATH+"label24.npy"),
# (125, BASE_AU_PATH+"image25.npy", BASE_AU_PATH+"label25.npy"),
# (126, BASE_AU_PATH+"image26.npy", BASE_AU_PATH+"label26.npy"),
# (127, BASE_AU_PATH+"image27.npy", BASE_AU_PATH+"label27.npy"),
# (128, BASE_AU_PATH+"image28.npy", BASE_AU_PATH+"label28.npy"),
# (129, BASE_AU_PATH+"image29.npy", BASE_AU_PATH+"label29.npy"),
# (130, BASE_AU_PATH+"image30.npy", BASE_AU_PATH+"label30.npy"),
# (131, BASE_AU_PATH+"image31.npy", BASE_AU_PATH+"label31.npy"),
# (132, BASE_AU_PATH+"image32.npy", BASE_AU_PATH+"label32.npy"),
# (133, BASE_AU_PATH+"image33.npy", BASE_AU_PATH+"label33.npy"),
# (134, BASE_AU_PATH+"image34.npy", BASE_AU_PATH+"label34.npy"),
# (135, BASE_AU_PATH+"image35.npy", BASE_AU_PATH+"label35.npy"),
# (136, BASE_AU_PATH+"image36.npy", BASE_AU_PATH+"label36.npy"),
# (137, BASE_AU_PATH+"image37.npy", BASE_AU_PATH+"label37.npy"),
# (138, BASE_AU_PATH+"image38.npy", BASE_AU_PATH+"label38.npy"),
# (139, BASE_AU_PATH+"image39.npy", BASE_AU_PATH+"label39.npy"),
# (140, BASE_AU_PATH+"image40.npy", BASE_AU_PATH+"label40.npy"),
# (141, BASE_AU_PATH+"image41.npy", BASE_AU_PATH+"label41.npy"),
# (142, BASE_AU_PATH+"image42.npy", BASE_AU_PATH+"label42.npy"),
# (143, BASE_AU_PATH+"image43.npy", BASE_AU_PATH+"label43.npy"),
# (144, BASE_AU_PATH+"image44.npy", BASE_AU_PATH+"label44.npy"),
# (145, BASE_AU_PATH+"image45.npy", BASE_AU_PATH+"label45.npy"),
# (146, BASE_AU_PATH+"image46.npy", BASE_AU_PATH+"label46.npy"),
# (147, BASE_AU_PATH+"image47.npy", BASE_AU_PATH+"label47.npy"),
# (148, BASE_AU_PATH+"image48.npy", BASE_AU_PATH+"label48.npy"),
# (149, BASE_AU_PATH+"image49.npy", BASE_AU_PATH+"label49.npy"),
# (150, BASE_AU_PATH+"image50.npy", BASE_AU_PATH+"label50.npy"),
# (151, BASE_AU_PATH+"image51.npy", BASE_AU_PATH+"label51.npy"),
# (152, BASE_AU_PATH+"image52.npy", BASE_AU_PATH+"label52.npy"),
# (153, BASE_AU_PATH+"image53.npy", BASE_AU_PATH+"label53.npy"),
# (154, BASE_AU_PATH+"image54.npy", BASE_AU_PATH+"label54.npy"),
# (155, BASE_AU_PATH+"image55.npy", BASE_AU_PATH+"label55.npy"),
# (156, BASE_AU_PATH+"image56.npy", BASE_AU_PATH+"label56.npy"),
# (157, BASE_AU_PATH+"image57.npy", BASE_AU_PATH+"label57.npy"),
# (158, BASE_AU_PATH+"image58.npy", BASE_AU_PATH+"label58.npy"),
# (159, BASE_AU_PATH+"image59.npy", BASE_AU_PATH+"label59.npy"),
# (160, BASE_AU_PATH+"image60.npy", BASE_AU_PATH+"label60.npy"),
# (161, BASE_AU_PATH+"image61.npy", BASE_AU_PATH+"label61.npy"),
# (162, BASE_AU_PATH+"image62.npy", BASE_AU_PATH+"label62.npy"),
# (163, BASE_AU_PATH+"image63.npy", BASE_AU_PATH+"label63.npy"),
# (164, BASE_AU_PATH+"image64.npy", BASE_AU_PATH+"label64.npy"),
# (165, BASE_AU_PATH+"image65.npy", BASE_AU_PATH+"label65.npy"),
# (166, BASE_AU_PATH+"image66.npy", BASE_AU_PATH+"label66.npy"),
# (167, BASE_AU_PATH+"image67.npy", BASE_AU_PATH+"label67.npy"),
# (168, BASE_AU_PATH+"image68.npy", BASE_AU_PATH+"label68.npy"),
# (169, BASE_AU_PATH+"image69.npy", BASE_AU_PATH+"label69.npy"),
# (170, BASE_AU_PATH+"image70.npy", BASE_AU_PATH+"label70.npy"),
# (171, BASE_AU_PATH+"image71.npy", BASE_AU_PATH+"label71.npy"),
# (172, BASE_AU_PATH+"image72.npy", BASE_AU_PATH+"label72.npy"),
# (173, BASE_AU_PATH+"image73.npy", BASE_AU_PATH+"label73.npy"),
# (174, BASE_AU_PATH+"image74.npy", BASE_AU_PATH+"label74.npy"),
# (175, BASE_AU_PATH+"image75.npy", BASE_AU_PATH+"label75.npy"),
# (176, BASE_AU_PATH+"image76.npy", BASE_AU_PATH+"label76.npy"),
# (177, BASE_AU_PATH+"image77.npy", BASE_AU_PATH+"label77.npy"),
# (178, BASE_AU_PATH+"image78.npy", BASE_AU_PATH+"label78.npy"),
# (179, BASE_AU_PATH+"image79.npy", BASE_AU_PATH+"label79.npy"),
# (180, BASE_AU_PATH+"image80.npy", BASE_AU_PATH+"label80.npy")]

BASE_AU_1_PATH = "/mnt/data/student/3Dircadb1/augurment_1"
train_dataset = [\
(301, IRCA_NUMPY_BASE_PATH + "image01.npy", IRCA_NUMPY_BASE_PATH + "label01.npy"),
(410, BASE_AU_1_PATH+"image01_01_00.npy", BASE_AU_PATH+"label01_01_00.npy"),
(411, BASE_AU_1_PATH+"image01_01_01.npy", BASE_AU_PATH+"label01_01_01.npy"),
(412, BASE_AU_1_PATH+"image01_01_02.npy", BASE_AU_PATH+"label01_01_02.npy"),
(413, BASE_AU_1_PATH+"image01_01_03.npy", BASE_AU_PATH+"label01_01_03.npy"),
(414, BASE_AU_1_PATH+"image01_01_04.npy", BASE_AU_PATH+"label01_01_04.npy"),
(415, BASE_AU_1_PATH+"image01_01_05.npy", BASE_AU_PATH+"label01_01_05.npy"),
(416, BASE_AU_1_PATH+"image01_01_06.npy", BASE_AU_PATH+"label01_01_06.npy"),
(417, BASE_AU_1_PATH+"image01_01_07.npy", BASE_AU_PATH+"label01_01_07.npy"),
(418, BASE_AU_1_PATH+"image01_01_08.npy", BASE_AU_PATH+"label01_01_08.npy"),
(419, BASE_AU_1_PATH+"image01_01_09.npy", BASE_AU_PATH+"label01_01_09.npy"),
(420, BASE_AU_1_PATH+"image01_02_00.npy", BASE_AU_PATH+"label01_02_00.npy"),
(421, BASE_AU_1_PATH+"image01_02_01.npy", BASE_AU_PATH+"label01_02_01.npy"),
(422, BASE_AU_1_PATH+"image01_02_02.npy", BASE_AU_PATH+"label01_02_02.npy"),
(423, BASE_AU_1_PATH+"image01_02_03.npy", BASE_AU_PATH+"label01_02_03.npy"),
(424, BASE_AU_1_PATH+"image01_02_04.npy", BASE_AU_PATH+"label01_02_04.npy"),
(425, BASE_AU_1_PATH+"image01_02_05.npy", BASE_AU_PATH+"label01_02_05.npy"),
(426, BASE_AU_1_PATH+"image01_02_06.npy", BASE_AU_PATH+"label01_02_06.npy"),
(427, BASE_AU_1_PATH+"image01_02_07.npy", BASE_AU_PATH+"label01_02_07.npy"),
(428, BASE_AU_1_PATH+"image01_02_08.npy", BASE_AU_PATH+"label01_02_08.npy"),
(429, BASE_AU_1_PATH+"image01_02_09.npy", BASE_AU_PATH+"label01_02_09.npy"),
(430, BASE_AU_1_PATH+"image01_03_00.npy", BASE_AU_PATH+"label01_03_00.npy"),
(431, BASE_AU_1_PATH+"image01_03_01.npy", BASE_AU_PATH+"label01_03_01.npy"),
(432, BASE_AU_1_PATH+"image01_03_02.npy", BASE_AU_PATH+"label01_03_02.npy"),
(433, BASE_AU_1_PATH+"image01_03_03.npy", BASE_AU_PATH+"label01_03_03.npy"),
(434, BASE_AU_1_PATH+"image01_03_04.npy", BASE_AU_PATH+"label01_03_04.npy"),
(435, BASE_AU_1_PATH+"image01_03_05.npy", BASE_AU_PATH+"label01_03_05.npy"),
(436, BASE_AU_1_PATH+"image01_03_06.npy", BASE_AU_PATH+"label01_03_06.npy"),
(437, BASE_AU_1_PATH+"image01_03_07.npy", BASE_AU_PATH+"label01_03_07.npy"),
(438, BASE_AU_1_PATH+"image01_03_08.npy", BASE_AU_PATH+"label01_03_08.npy"),
(439, BASE_AU_1_PATH+"image01_03_09.npy", BASE_AU_PATH+"label01_03_09.npy"),
(440, BASE_AU_1_PATH+"image01_04_00.npy", BASE_AU_PATH+"label01_04_00.npy"),
(441, BASE_AU_1_PATH+"image01_04_01.npy", BASE_AU_PATH+"label01_04_01.npy"),
(442, BASE_AU_1_PATH+"image01_04_02.npy", BASE_AU_PATH+"label01_04_02.npy"),
(443, BASE_AU_1_PATH+"image01_04_03.npy", BASE_AU_PATH+"label01_04_03.npy"),
(444, BASE_AU_1_PATH+"image01_04_04.npy", BASE_AU_PATH+"label01_04_04.npy"),
(445, BASE_AU_1_PATH+"image01_04_05.npy", BASE_AU_PATH+"label01_04_05.npy"),
(446, BASE_AU_1_PATH+"image01_04_06.npy", BASE_AU_PATH+"label01_04_06.npy"),
(447, BASE_AU_1_PATH+"image01_04_07.npy", BASE_AU_PATH+"label01_04_07.npy"),
(448, BASE_AU_1_PATH+"image01_04_08.npy", BASE_AU_PATH+"label01_04_08.npy"),
(449, BASE_AU_1_PATH+"image01_04_09.npy", BASE_AU_PATH+"label01_04_09.npy"),
(450, BASE_AU_1_PATH+"image01_05_00.npy", BASE_AU_PATH+"label01_05_00.npy"),
(451, BASE_AU_1_PATH+"image01_05_01.npy", BASE_AU_PATH+"label01_05_01.npy"),
(452, BASE_AU_1_PATH+"image01_05_02.npy", BASE_AU_PATH+"label01_05_02.npy"),
(453, BASE_AU_1_PATH+"image01_05_03.npy", BASE_AU_PATH+"label01_05_03.npy"),
(454, BASE_AU_1_PATH+"image01_05_04.npy", BASE_AU_PATH+"label01_05_04.npy"),
(455, BASE_AU_1_PATH+"image01_05_05.npy", BASE_AU_PATH+"label01_05_05.npy"),
(456, BASE_AU_1_PATH+"image01_05_06.npy", BASE_AU_PATH+"label01_05_06.npy"),
(457, BASE_AU_1_PATH+"image01_05_07.npy", BASE_AU_PATH+"label01_05_07.npy"),
(458, BASE_AU_1_PATH+"image01_05_08.npy", BASE_AU_PATH+"label01_05_08.npy"),
(459, BASE_AU_1_PATH+"image01_05_09.npy", BASE_AU_PATH+"label01_05_09.npy"),
(460, BASE_AU_1_PATH+"image01_06_00.npy", BASE_AU_PATH+"label01_06_00.npy"),
(461, BASE_AU_1_PATH+"image01_06_01.npy", BASE_AU_PATH+"label01_06_01.npy"),
(462, BASE_AU_1_PATH+"image01_06_02.npy", BASE_AU_PATH+"label01_06_02.npy"),
(463, BASE_AU_1_PATH+"image01_06_03.npy", BASE_AU_PATH+"label01_06_03.npy"),
(464, BASE_AU_1_PATH+"image01_06_04.npy", BASE_AU_PATH+"label01_06_04.npy"),
(465, BASE_AU_1_PATH+"image01_06_05.npy", BASE_AU_PATH+"label01_06_05.npy"),
(466, BASE_AU_1_PATH+"image01_06_06.npy", BASE_AU_PATH+"label01_06_06.npy"),
(467, BASE_AU_1_PATH+"image01_06_07.npy", BASE_AU_PATH+"label01_06_07.npy"),
(468, BASE_AU_1_PATH+"image01_06_08.npy", BASE_AU_PATH+"label01_06_08.npy"),
(469, BASE_AU_1_PATH+"image01_06_09.npy", BASE_AU_PATH+"label01_06_09.npy")]

test_dataset = [\
(315, IRCA_NUMPY_BASE_PATH+"image15.npy",IRCA_NUMPY_BASE_PATH+"label15.npy"),
(316, IRCA_NUMPY_BASE_PATH+"image16.npy",IRCA_NUMPY_BASE_PATH+"label16.npy"),
(317, IRCA_NUMPY_BASE_PATH+"image17.npy",IRCA_NUMPY_BASE_PATH+"label17.npy"),
(318, IRCA_NUMPY_BASE_PATH+"image18.npy",IRCA_NUMPY_BASE_PATH+"label18.npy"),
(319, IRCA_NUMPY_BASE_PATH+"image19.npy",IRCA_NUMPY_BASE_PATH+"label19.npy"),
(320, IRCA_NUMPY_BASE_PATH+"image20.npy",IRCA_NUMPY_BASE_PATH+"label20.npy")]
