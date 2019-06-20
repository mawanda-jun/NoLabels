import os
from typing import List
import random
import h5py
import numpy as np
from PIL import Image, ImageFile
import threading
from labels import labels_dict
# force pillow to load also truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ThreadedH5pyFile(threading.Thread):
    """
    Threaded version to prepare the dataset. Everything runs smoothly because we have multiple labels that avoid
    race conditions
    """
    def __init__(self, img_list, set_type, hdf5_out, img_size):
        threading.Thread.__init__(self)
        self.img_list = img_list
        self.set_type = set_type
        self.hdf5_out = hdf5_out
        self.img_size = img_size
        self.errors = set([])
        self.training_mean_new = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        self.training_mean_old = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        self.training_variance = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

    def run(self):
        # TODO: inserire crop immagine come descritto nel paper sezione 3.4 righe 3-5
        for i, label_and_path in enumerate(self.img_list):
            #  label_and_path contains the label and the path_to_image. It's a tuple
            if i % 1000 == 0:
                print('\n-------------\nThread: {name}\nSaved images: {c}/{tot}'.format(
                   name=self.set_type,
                   c=i,
                   tot=len(self.img_list)
                ))
            try:
                # some images are not well formatted or have bytes error. So we try to open them and, in case of error,
                # keep the path inside the array self.errors
                img = Image.open(label_and_path[1])
                # in case of b/w images it doesn't breaks the line
                if img.mode == 'RGB':
                    img = img.resize((self.img_size, self.img_size), resample=Image.LANCZOS)
                    # convert pillow image into a np array
                    np_img = np.array(img)
                    # calculate per feature mean and variance on training data only
                    if self.set_type == 'train':
                        # Welford method for online calculation of mean and variance
                        if i > 0:
                            self.training_mean_new = self.training_mean_old + \
                                                (np_img - self.training_mean_old) / (i+1)

                            self.training_variance = self.training_variance + \
                                                (np_img - self.training_mean_old) * \
                                                (np_img - self.training_mean_new)
                            self.training_mean_old = self.training_mean_new
                        else:
                            self.training_mean_old = np.array(np_img, dtype=np.float32)
                    else:
                        self.training_mean_new = None
                        self.training_mean_old = None
                        self.training_variance = None
                    # write images inside h5 file
                    self.hdf5_out['X_' + self.set_type][i, ...] = np_img
                    self.hdf5_out['Y_' + self.set_type][i, ...] = labels_dict[label_and_path[0]]
            except OSError as e:
                self.errors.add(str(e) + '\n')
                continue
        print('THREAD {} HAS FINISHED'.format(self.set_type))


def images_paths(folder_path: str) -> List[str]:
    """
    Collects all images from one folder and return a list of paths
    :param folder_path:
    :return:
    """
    paths = []
    folder_path = os.path.join(os.getcwd(), folder_path)
    extensions = set([])
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            extensions.add(os.path.splitext(file)[1])
            paths.append(os.path.join(root, file))
    with open('extensions.txt', 'w') as f:
        f.writelines([extension + '\n' for extension in extensions])
    return paths


def apply_label(lst: List[str], label: str) -> List:
    """
    Apply label to all images in path. Returns a list of tuple (label, path_to_image)
    :param lst:
    :param label:
    :return:
    """
    return [(label, path) for path in lst]


def read_and_label(labels: List[str], dirpath: str = 'images') -> List:
    """
    Read all images in folders named the same way as labels. Returns a list of all images with their labels
    :param dirpath:
    :param labels:
    :return:
    """
    dataset = {}

    for label in labels:
        dataset[label] = apply_label(images_paths(os.path.join(dirpath, label)), label)
    global_array = []

    for _, paths in dataset.items():
        global_array.extend(paths)
    return global_array


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


def generate_h5py(file_list: List, img_size=256, hdf5_file_name: str = 'data', train_dim: float = 0.65, val_dim: float = 0.25):
    """
    Generate and save train, validation and test data. Test data is what is left from train and validation sets
    :param file_list:
    :param img_size:
    :param train_dim:
    :param val_dim:
    :param hdf5_file_name:
    :return:
    """
    shuffle_dataset(file_list)
    # make train, evaluation and test partitions
    n = len(file_list)
    train_i = [0, int(train_dim*n)]
    eval_i = [int(train_dim*n) + 1, int((train_dim+val_dim)*n)]
    test_i = [int((train_dim+val_dim)*n)+1, -1]
    file_dict = {
        'train': file_list[train_i[0]:train_i[1]],
        'eval': file_list[eval_i[0]:eval_i[1]],
        'test': file_list[test_i[0]:test_i[1]]
    }

    # it is better to keep evaluation dataset bigger than test one
    assert len(file_dict['train']) > len(file_dict['eval']) > len(file_dict['test'])

    # create dataset parts: X_<set_type> will contain the images, the Y_<set_type> the labels,
    # according to the labels of labels_dict
    hdf5_out = h5py.File(hdf5_file_name, mode='w')
    hdf5_out.create_dataset(name='X_train', shape=(len(file_dict['train']), img_size, img_size, 3), dtype=np.uint8)
    hdf5_out.create_dataset('Y_train', (len(file_dict['train']), 1), np.uint8)
    hdf5_out.create_dataset('X_eval', (len(file_dict['eval']), img_size, img_size, 3), np.uint8)
    hdf5_out.create_dataset('Y_eval', (len(file_dict['eval']), 1), np.uint8)
    hdf5_out.create_dataset('X_test', (len(file_dict['test']), img_size, img_size, 3), np.uint8)
    hdf5_out.create_dataset('Y_test', (len(file_dict['test']), 1), np.uint8)
    hdf5_out.create_dataset('train_mean', (img_size, img_size, 3), np.float32)
    hdf5_out.create_dataset('train_std', (img_size, img_size, 3), np.float32)

    # make one thread for <set_type>
    threaded_types = []
    for set_type, img_list in file_dict.items():
        threaded_types.append(ThreadedH5pyFile(img_list, set_type, hdf5_out, img_size))

    for thread in threaded_types:
        thread.start()

    for thread in threaded_types:
        # wait for the threads to finish the execution
        thread.join()

    for i, thread in enumerate(threaded_types):
        if thread.errors is not []:
            with open('errors{}.txt'.format(i), 'w') as f:
                f.writelines(thread.errors)
        if thread.training_variance is not None:
            # calculate the std using the variace array only for train set
            training_std = np.sqrt(thread.training_variance / (len(file_dict['train']) - 1))
            hdf5_out['train_mean'][...] = thread.training_mean_new
            hdf5_out['train_std'][...] = training_std
    hdf5_out.close()


if __name__ == '__main__':
    labels = ['cats', 'dogs', 'fishes']

    images_list = read_and_label(labels)
    generate_h5py(images_list, 256, 'dataset.h5')
