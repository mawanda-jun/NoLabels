"""
This file transform the target task dataset into one that is feasible for our FT class.
Every file is counted, divided between train, validation and test set, cropped and resized.
Mean and STD are calculated for training set
"""
import os
from typing import List
import random
import h5py
import numpy as np
from PIL import Image, ImageFile
import threading
# force pillow to load also truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# number of images to take from the folder
N_EL = int(5e5)
# path/to/folder that contains the images. Every image will be rewritten inside output_path with the name of the
# original containing folder
RES_PATH = os.path.join('Dataset/resources/images/food101')


def square_img(im: Image.Image) -> Image:
    """

    :param im:
    :return:
    """
    w, h = im.size
    if w == h:
        return im
    crop_shift = abs(h-w) // 2  # central crop
    if w > h:
        # left-upper, right-lower
        # box dimension must be that way
        box = [0, 0, h, h]
        # and it may be moved horizontally
        box[0] += crop_shift
        box[2] += crop_shift
    else:
        # moving box vertically
        box = [0, 0, w, w]
        box[1] += crop_shift
        box[3] += crop_shift
    im = im.crop(box)
    return im


class ThreadedImageWriter(threading.Thread):
    """
    Threaded version to prepare the dataset. Everything runs smoothly because we have multiple folders that avoid
    race conditions
    """
    def __init__(self, img_list: List[str], set_type: str, hdf5_out: h5py.File, img_size: int, dataset_folder: str):
        threading.Thread.__init__(self)
        self.img_list = img_list
        self.set_type = set_type
        self.hdf5_out = hdf5_out
        self.img_size = img_size
        self.img_folder = os.path.join(dataset_folder, set_type)
        os.makedirs(self.img_folder, exist_ok=True)
        self.read_errors = set([])
        self.mean = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        self.M2 = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

    def run(self):
        for i, path in enumerate(self.img_list):
            #  path contains the path/to/image
            if i % 1000 == 0:
                print('\n-------------\nThread: {name}\nSaved images: {c}/{tot}'.format(
                   name=self.set_type,
                   c=i,
                   tot=len(self.img_list)
                ))
            try:
                img = Image.open(path)
                img = square_img(img)
                img = img.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
                dirname = os.path.dirname(path).split(os.sep)[-1]

                img.save(os.path.join(self.img_folder, str(dirname) + '%' + str(i)+'.jpg'))
                # convert pillow image into a np array
                np_img = np.array(img)
                # calculate per feature mean and variance on training data only
                if self.set_type == 'train':
                    # Welford method for online calculation of mean and variance
                    # when i==0 the step is useless since not mean nor M2 changes. However this is less heavy than
                    # check i > 0 every iteration.
                    try:
                        delta = np.subtract(np_img, self.mean)
                    except ValueError:
                        # in case of grayscale images we have to make the image as a RGB one
                        np_img = np.expand_dims(np_img, axis=-1)
                        np_img = np.concatenate([np_img]*3, axis=-1)
                        # calculate delta again
                        delta = np.subtract(np_img, self.mean)
                    self.mean = np.add(self.mean, np.divide(delta, (i + 1)))
                    self.M2 = np.add(self.M2, np.multiply(delta, np.subtract(np_img, self.mean)))
            except OSError:
                raise ValueError("image {} has problems".format(path))
        print('THREAD {} HAS FINISHED'.format(self.set_type))


def images_in_paths(folder_path: str) -> List[str]:
    """
    Collects all images from one folder and return a list of paths
    :param folder_path:
    :return:
    """
    paths = []
    folder_path = os.path.join(os.getcwd(), folder_path)
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            paths.append(os.path.join(root, file))
    return paths


def shuffle_dataset(lst: List, seed: int = None) -> None:
    """
    Controlled shuffle.
    :param lst:
    :param seed: if specified the shuffle returns the same shuffled list every time it is invoked
    :return:
    """
    if seed is not None:
        random.seed(seed)
    random.shuffle(lst)


def generate_dataset(file_list: List, dataset_folder: str, img_size=225, train_dim: float = 0.65, val_dim: float = 0.25):
    """
    Generate and save train, validation and test data. Test data is what is left from train and validation sets
    :param file_list:
    :param img_size:
    :param train_dim:
    :param val_dim:
    :return:
    """
    shuffle_dataset(file_list)
    # make train, validation and test partitions
    n = len(file_list)
    train_i = [0, int(train_dim*n)]
    val_i = [int(train_dim*n), int((train_dim+val_dim)*n)]
    test_i = [int((train_dim+val_dim)*n), -1]
    file_dict = {
        'train': file_list[train_i[0]:train_i[1]],
        'val': file_list[val_i[0]:val_i[1]],
        'test': file_list[test_i[0]:]
    }

    # it is better to keep validation dataset bigger than test one
    assert len(file_dict['train']) > len(file_dict['val']) > len(file_dict['test'])

    os.makedirs(dataset_folder, exist_ok=True)

    # create h5file to store information about train_mean and train_std that are useful for training later
    h5_path = os.path.join(dataset_folder, 'info.h5')
    with h5py.File(h5_path, mode='w') as hdf5_out:
        hdf5_out.create_dataset('train_mean', (img_size, img_size, 3), np.float32)
        hdf5_out.create_dataset('train_std', (img_size, img_size, 3), np.float32)
        hdf5_out.create_dataset('train_dim', (), np.int32, data=int(n*train_dim))
        hdf5_out.create_dataset('val_dim', (), np.int32, data=int(n*val_dim))
        hdf5_out.create_dataset('test_dim', (), np.int32, data=int(n*(1-train_dim-val_dim)))
        # make one thread for <set_type>
        threaded_types = []
        for set_type, img_list in file_dict.items():
            threaded_types.append(ThreadedImageWriter(img_list, set_type, hdf5_out, img_size, dataset_folder))

        for thread in threaded_types:
            thread.start()

        for thread in threaded_types:
            # wait for the threads to finish the execution
            thread.join()

        for i, thread in enumerate(threaded_types):
            if thread.read_errors:
                with open('errors{}.txt'.format(i), 'w') as f:
                    f.writelines(thread.read_errors)
            if thread.set_type == 'train':
                # calculate the std using the variace array only for train set
                training_std = np.sqrt(thread.M2 / (len(file_dict['train']) - 1))
                hdf5_out['train_mean'][...] = thread.mean
                hdf5_out['train_std'][...] = training_std


if __name__ == '__main__':
    output_path = os.path.join(os.getcwd(), 'resources', 'images')
    # elements = N_EL
    res_path = RES_PATH
    images_list = images_in_paths(os.path.join(res_path))
    random.shuffle(images_list)
    # images_list = images_list[0:elements]
    generate_dataset(images_list, os.path.join(output_path, 'food101_resized'))
