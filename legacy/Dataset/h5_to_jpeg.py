import os
from typing import List
import random
import h5py
import numpy as np
from PIL import Image, ImageFile
import threading
# force pillow to load also truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ThreadedH5pyToJPEG(threading.Thread):
    """
    Threaded version to prepare the dataset. Everything runs smoothly because we have multiple labels that avoid
    race conditions
    """
    def __init__(self, set, type, dest_folder):
        threading.Thread.__init__(self)
        self.set = set
        self.type = type
        self.dest_folder = dest_folder

    def run(self):
        folder = os.path.join(dest_folder, str(self.type))
        os.makedirs(folder, exist_ok=True)
        for i, arr in enumerate(self.set):
            file = os.path.join(folder, self.type+str(i)+'.jpg')
            Image.fromarray(arr).save(file)


if __name__ == '__main__':
    h5_file = 'ILSVRC_5e5.h5'
    path_to_h5 = os.path.join(os.getcwd(), 'resources', 'h5_files', h5_file)
    dest_folder = os.path.join(os.getcwd(), 'resources', 'images', '5e5')
    with h5py.File(path_to_h5, 'a') as file:
        train_set = file['train_img']
        val_set = file['val_img']
        test_set = file['test_img']
        threads = [ThreadedH5pyToJPEG(train_set, 'train', dest_folder), ThreadedH5pyToJPEG(val_set, 'val', dest_folder), ThreadedH5pyToJPEG(test_set, 'test', dest_folder)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        del file['train_img']
        del file['val_img']
        del file['test_img']
