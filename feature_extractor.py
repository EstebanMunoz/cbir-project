import os
import pickle

import numpy as np

from utils import load_img
from utils import kwargs, extractors


def get_img_features(img: np.ndarray, feature_extractor: str, **kwargs) -> np.ndarray:
    if not feature_extractor in extractors.keys():
        raise ValueError(f"Expected str 'cnn' or 'handcrafted', got {feature_extractor}")
    
    extractor = extractors[feature_extractor]
    return extractor(img, **kwargs)


def get_dataset_features(dataset_path, feature_extractor, **kwargs):
    img_dirs = sorted(os.listdir(dataset_path))

    if feature_extractor == 'cnn':
        dataset_features = np.empty((len(img_dirs), 1024))
    else:
        dataset_features = np.empty((len(img_dirs), 128))

    for i, img_name in enumerate(img_dirs):
        img_path = os.path.join(dataset_path, img_name)
        img = load_img(img_path)
        img_features = get_img_features(img, feature_extractor, **kwargs)
        dataset_features[i] = img_features
        if (i+1) % 50 == 0:
            print(f"{i+1}/{len(img_dirs)} features extracted")
    print("All features have been extracted")
    
    return dataset_features, img_dirs


if __name__ == '__main__':
    holidays_dataset = 'src/holidays_dataset/database'
    GPR1200_dataset = 'src/GPR1200_dataset'
    datasets = {'holidays': holidays_dataset, 'GPR1200': GPR1200_dataset}

    for dataset, path in datasets.items():
        print(f'Extracting {dataset} cnn features')
        cnn_dataset_features, img_names = get_dataset_features(path, 'cnn', **kwargs)
        cnn_dataset_features = cnn_dataset_features.reshape(len(img_names), 1024)
        np.save(f'features/{dataset}_cnn_features', cnn_dataset_features)
        with open(f'features/{dataset}_img_names.pickle', 'wb') as dir:
            pickle.dump(img_names, dir)
        print(f'{dataset} cnn features extracted')


        print()
        print(f'Extracting {dataset} handcrafted features')
        handcrafted_dataset_features, img_names = get_dataset_features(path, 'handcrafted', **kwargs)    
        np.save(f'features/{dataset}_handcrafted_features', handcrafted_dataset_features)
        print(f'{dataset} handcrafted features extracted')
        print()
