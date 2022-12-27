import pickle
import os

import numpy as np

from utils import load_img
from utils import kwargs
from feature_extractor import get_img_features


def euclidean_distance(features, dataset_features):
    euclidean_dist = np.sqrt(np.power(dataset_features - features, 2).sum(axis=1))
    return euclidean_dist


def handcrafted_distance(features, dataset_features, alpha1, alpha2, eps):
    mid_idx = features.shape[0]//2
    ccf = features[:mid_idx]
    bpf = features[mid_idx:]

    dataset_ccf = dataset_features[:, :mid_idx]
    dataset_bpf = dataset_features[:, mid_idx:]

    ccf_distance = alpha1*np.abs(dataset_ccf - ccf)/(dataset_ccf + ccf + eps)
    bpf_distance = alpha2*np.abs(dataset_bpf - bpf)/(dataset_bpf + bpf + eps)

    return ccf_distance.sum(axis=1) + bpf_distance.sum(axis=1)


def same_class_holidays_dataset(img_name1, img_name2):
    img_class = slice(1, 4)
    return img_name1[img_class] == img_name2[img_class]


def same_class_GPR1200_dataset(img_name1, img_name2):
    img1_class = img_name1.split('_')[0]
    img2_class = img_name2.split('_')[0]
    return img1_class == img2_class


def same_class_imgs(img_name, img_dataset, dataset):
    if dataset == 'holidays':
        same_class = same_class_holidays_dataset
    elif dataset == 'GPR1200':
        same_class = same_class_GPR1200_dataset
    else:
        raise ValueError(f'''Expected str 'holidays' or 'GPR1200'. Got {dataset}.''')
    
    imgs = []
    for img in img_dataset:
        if same_class(img_name, img):
            imgs.append(img)
    return imgs


def retrieve_imgs(img_name, img_dir, dataset, feature_extractor):
    img = load_img(os.path.join(img_dir, img_name))
    features = get_img_features(img, feature_extractor, **kwargs)
    dataset_features = np.load(f'features/{dataset}_{feature_extractor}_features.npy')
    with open(f'features/{dataset}_img_names.pickle', 'rb') as data:
        img_names = pickle.load(data)
    
    # distances = handcrafted_distance(features, dataset_features, 1, 1, 0.0001)
    distances = euclidean_distance(features, dataset_features)
    sorted_args = np.argsort(distances)
    sorted_imgs = [img_names[i] for i in sorted_args]

    return sorted_imgs


def rank_score(img_name, img_dir, dataset, feature_extractor):
    sorted_imgs = retrieve_imgs(img_name, img_dir, dataset, feature_extractor)
    class_imgs = same_class_imgs(img_name, sorted_imgs, dataset)
    class_len = len(class_imgs)

    idxs = [sorted_imgs.index(img) for img in class_imgs]
    rank = sum(idx + 1 for idx in idxs)/class_len

    return rank


def normalized_rank_score(img_name, img_dir, dataset, feature_extractor):
    sorted_imgs = retrieve_imgs(img_name, img_dir, dataset, feature_extractor)
    class_imgs = same_class_imgs(img_name, sorted_imgs, dataset)
    
    class_len = len(class_imgs)
    N = len(sorted_imgs)

    idxs = [sorted_imgs.index(img) for img in class_imgs]
    rank = sum(idx + 1 for idx in idxs)

    return (rank - class_len*(class_len+1)/2)/(class_len*N)


def average_normalized_rank(query_dir, dataset, feature_extractor):
    query_imgs = os.listdir(query_dir)
    norm_ranks = np.zeros(len(query_imgs))

    for i, img in enumerate(query_imgs):
        norm_ranks[i] = normalized_rank_score(img, query_dir, dataset, feature_extractor)
    
    return norm_ranks.sum()/len(query_imgs)