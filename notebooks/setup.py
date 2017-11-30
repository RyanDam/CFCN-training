CAFE_MODE = 'GPU'
# CAFE_MODE = 'CPU'

############### TRAINING CONFIG ###############
PLOT_INTERVAL = 100
TEST_NUMBER = 300

# RUN_MODE = 'resume-last'
# STATE_FOLDER = '/mnt/data/student/snapshot/'
# RUN_MODE = 'resume'
# STATE_FILE = '/mnt/data/student/snapshot/_iter_9000.solverstate'
RUN_MODE = 'retrain'
PRE_TRAIN_WEIGHTS = 'phseg_v5.caffemodel'
# PRE_TRAIN_WEIGHTS = '/mnt/data/student/snapshot_step1_p1/_iter_6000.caffemodel'

MONITOR_FOLDER = 'monitor/%s'

############### TEST DICE, CHECKPOINT ###############
TEST_WEIGHT_FILE = '/mnt/data/student/snapshot/_iter_%d.caffemodel'
TEST_STATE_FOLDER = '/mnt/data/student/snapshot/'

############### INFERENCE ###############
STEP1_DEPLOY_PROTOTXT = "/mnt/data/student/deploy/step1_deploy.prototxt"
STEP1_MODEL_WEIGHTS   = "/mnt/data/student/deploy/our/step1_weights.caffemodel"
STEP2_DEPLOY_PROTOTXT = "/mnt/data/student/deploy/step2_deploy.prototxt"
STEP2_MODEL_WEIGHTS   = "/mnt/data/student/deploy/our/step2_weights.caffemodel"

PATIENT_DICOM_PATH = "/mnt/data/student/3Dircadb1/3Dircadb1.19/PATIENT_DICOM_NORM"
PATIENT_MASH_PATH = "/mnt/data/student/3Dircadb1/3Dircadb1.19/MASKS_DICOM/"
C_MIN_THRESHOLD = 800
C_MAX_THRESHOLD = 1400
INFERENCE_SAVE_FOLDER = '/mnt/data/student/inference/%s'
