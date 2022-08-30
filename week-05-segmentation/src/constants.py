from os import path as osp


PROJECT_PATH = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../'))
DATA_PATH = osp.join(PROJECT_PATH, 'data')
TRAIN_IMAGES_PATH = osp.join(DATA_PATH, 'train_images')
EXPERIMENTS_PATH = osp.join(PROJECT_PATH, 'experiments')
