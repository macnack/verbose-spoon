# configs/data/synthetic_homography.py

from configs.data.base import cfg

# Assume your images are in 'data/coco/train2017' and you have a list file.
# You can create this list file with: ls data/coco/train2017/ > data/coco/train_list.txt
TRAIN_IMAGE_ROOT = "/home/mackop/workspace/data/my_pregen_dataset2/train"
# VAL_IMAGE_ROOT = "/home/mackop/workspace/data/my_pregen_dataset2/val"
TRAIN_IMG_SIZE = 3
VAL_IMG_SIZE = 1
IGNORE_FILE = None
cfg.DATASET.TRAINVAL_DATA_SOURCE = "synthetichomographypregen"

cfg.DATASET.TRAIN_DATA_ROOT = TRAIN_IMAGE_ROOT
cfg.DATASET.TRAIN_NUM_SAMPLES = TRAIN_IMG_SIZE
cfg.DATASET.VAL_NUM_SAMPLES = VAL_IMG_SIZE
cfg.DATASET.IGNORE_LIST = IGNORE_FILE

# You can optionally create a validation set as well
cfg.DATASET.VAL_DATA_ROOT = TRAIN_IMAGE_ROOT

# Set training image resolution
IMG_H, IMG_W = 480, 640
cfg.EDM.TRAIN_RES_H = IMG_H
cfg.EDM.TRAIN_RES_W = IMG_W
cfg.EDM.TEST_RES_H = IMG_H
cfg.EDM.TEST_RES_W = IMG_W

cfg.EDM.NECK.NPE = [
    cfg.EDM.TRAIN_RES_H,
    cfg.EDM.TRAIN_RES_W,
    cfg.EDM.TEST_RES_H,
    cfg.EDM.TEST_RES_W,
]