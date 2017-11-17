
CAFE_MODE = 'GPU'
# CAFE_MODE = 'CPU'

############### TRAINING CONFIG ###############
PLOT_INTERVAL = 100

# RUN_MODE = 'resume-last'
# STATE_FOLDER = '/mnt/data/student/snapshot/'
# RUN_MODE = 'resume'
# STATE_FILE = '/mnt/data/student/snapshot/_iter_23000.solverstate'
RUN_MODE = 'retrain'
PRE_TRAIN_WEIGHTS = 'phseg_v5.caffemodel'

MONITOR_FOLDER = 'monitor/%s'

############### TEST DICE, CHECKPOINT ###############
TEST_WEIGHT_FILE = '/mnt/data/student/snapshot/_iter_%d.caffemodel'
TEST_STATE_FOLDER = '/mnt/data/student/snapshot/'

############### INFERENCE ###############
STEP1_DEPLOY_PROTOTXT = "inference/step1_deploy.prototxt"
# STEP1_MODEL_WEIGHTS   = "inference/step1_weights.caffemodel"
# STEP1_MODEL_WEIGHTS   = "/mnt/data/student/snapshot_save/snapshot_step1_retrain/_iter_38000.caffemodel"
# STEP1_MODEL_WEIGHTS   = "/mnt/data/student/snapshot_step1/_iter_2500.caffemodel"
STEP1_MODEL_WEIGHTS   = "/mnt/data/student/snapshot_step1_enhanced/_iter_8500.caffemodel"
STEP2_DEPLOY_PROTOTXT = "inference/step2_deploy.prototxt"
# STEP2_MODEL_WEIGHTS   = "inference/step2_weights.caffemodel"
STEP2_MODEL_WEIGHTS   = "/mnt/data/student/snapshot_step2/_iter_23000.caffemodel"
# STEP2_MODEL_WEIGHTS   = "/mnt/data/student/snapshot/_iter_3500.caffemodel"

PATIENT_DICOM_PATH = "test_image/3Dircadb1.18/PATIENT_DICOM/"
PATIENT_MASH_PATH = "test_image/3Dircadb1.18/MASKS_DICOM/"
C_MIN_THRESHOLD = 800
C_MAX_THRESHOLD = 1400
INFERENCE_SAVE_FOLDER = 'output/%s'