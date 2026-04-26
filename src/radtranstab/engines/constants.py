

# Dataloader Settings 

TRAIN_RADIOMCIS_FILE = "./datasets/train_radiomcis.json"
VAL_RADIOMCIS_FILE = "./datasets/val_radiomcis.json"
FEATURE_LIST_PATH = "./datasets/radiomics_features_list.json"
ROOT_DIR = "./datasets/"

LABEL_COL = "target_label"
ID_COL = "barcode"


# Model Settings
# commented out is same as default in build_radiomics_learner, so we can directly use the default values without passing them as arguments

CHECKPOINT_PATH = "./checkpoint/"
OUTPUT_DIR = "./output"

NUM_CLASS = 5
# HIDDEN_DIM = 128
# NUM_LAYER = 2
PROJECTION_DIM = 384
DROPOUT = 0.1
ACTIVATION = "leakyrelu"
# NUM_SUB_COLS = [72, 54, 36, 18, 9, 3, 1]    
APE_DROP_RATE = 0.0


# Train Settings 

SEED = 0
DEVICE = "cuda:0"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 4
USE_AMP = False