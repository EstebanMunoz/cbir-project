import os

import cv2
import numpy as np

from cnn_model import extract_features as cnn_feature_extractor
from handcrafted_model import extract_features as handcrafted_feature_extractor


COLOR_BLOCK_SHAPE = (8,8,1)
GRAY_BLOCK_SHAPE = (8,8)
CODEBOOK_SIZE = 64
KERNEL = np.array([[0,0,7], [3,5,1]])/16
RESIZE = 512
SEED = 1

kwargs = {
'color_block_shape': COLOR_BLOCK_SHAPE,
'gray_block_shape': GRAY_BLOCK_SHAPE,
'codebook_size': CODEBOOK_SIZE,
'kernel': KERNEL,
'resize': RESIZE,
'seed': SEED
}

extractors = {
    'cnn': cnn_feature_extractor,
    'handcrafted': handcrafted_feature_extractor
}


def load_img(path: str) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def query_imgs_list(source: str):
    imgs = sorted(os.listdir(source))
    return [img for img in imgs if img.endswith('00.jpg')]


def move_imgs(source, destination):
    query_imgs = query_imgs_list(source)
    for img in query_imgs:
        os.rename(os.path.join(source, img), os.path.join(destination, img))