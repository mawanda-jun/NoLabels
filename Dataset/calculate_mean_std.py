import os
import numpy as np
from PIL import Image
from typing import List
import h5py
from tqdm import tqdm

image_size = 256
channels = 3


def online_variance(paths):
    # n = 0 == i+1
    mean = np.zeros((image_size, image_size, channels))
    M2 = np.zeros((image_size, image_size, channels))
    i = 0
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        np_img = np.array(Image.open(path))
        delta = np.subtract(np_img, mean)
        mean = np.add(mean, np.divide(delta, (i+1)))
        M2 = np.add(M2, np.multiply(delta, np.subtract(np_img, mean)))

    return mean, np.sqrt(np.divide(M2, i))


def images_in_paths(folder_path: str) -> List[str]:
    """
    Collects paths to all images from one folder and return them as a list
    :param folder_path:
    :return: list of path/to/image
    """
    paths = []
    folder_path = os.path.join(os.getcwd(), folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


if __name__ == '__main__':
    folder = "C:\\Users\\giova\\Downloads\\Dataset_NoLabels\\ILSVRC_10000"
    train_images = images_in_paths(os.path.join(folder, 'train'))
    val_images = images_in_paths(os.path.join(folder, 'val'))
    test_images = images_in_paths(os.path.join(folder, 'test'))
    mean, std = online_variance(train_images)

    with h5py.File(os.path.join(folder, 'info.h5'), mode='w') as hdf5_out:
        hdf5_out.create_dataset('train_mean', (image_size, image_size, 3), np.float32, data=mean)
        hdf5_out.create_dataset('train_std', (image_size, image_size, 3), np.float32, data=std)
        hdf5_out.create_dataset('train_dim', (), np.int32, data=len(train_images))
        hdf5_out.create_dataset('val_dim', (), np.int32, data=len(val_images))
        hdf5_out.create_dataset('test_dim', (), np.int32, data=len(test_images))