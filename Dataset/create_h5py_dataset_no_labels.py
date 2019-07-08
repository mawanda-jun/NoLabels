import os
from typing import List
import random
import h5py
import numpy as np
from PIL import Image, ImageFile
import threading
# force pillow to load also truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def square_img(im: Image.Image) -> Image:
    """

    :param im:
    :return:
    """
    w, h = im.size
    if w == h:
        return im
    crop_shift = random.randrange(abs(h-w))  # crops only in the dimension that is bigger!
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


class ThreadedH5pyFile(threading.Thread):
    """
    Threaded version to prepare the dataset. Everything runs smoothly because we have multiple labels that avoid
    race conditions
    """
    def __init__(self, img_list: List[str], set_type: str, hdf5_out: h5py.File, img_size: int):
        threading.Thread.__init__(self)
        self.img_list = img_list
        self.set_type = set_type
        self.hdf5_out = hdf5_out
        self.img_size = img_size
        self.read_errors = set([])
        self.training_mean_new = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        self.training_mean_old = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
        self.training_variance = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)

    def run(self):
        for i, path in enumerate(self.img_list):
            #  path contains the label and the path_to_image. It's a tuple
            if i % 1000 == 0:
                print('\n-------------\nThread: {name}\nSaved images: {c}/{tot}'.format(
                   name=self.set_type,
                   c=i,
                   tot=len(self.img_list)
                ))
            try:
                # some images are not well formatted or have bytes error. So we try to open them and, in case of error,
                # keep the path inside the array self.errors
                img = Image.open(path)
                # in case of b/w images it doesn't breaks the line
                if img.mode == 'RGB':
                    img = square_img(img)
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
                    self.hdf5_out[self.set_type + '_img'][i, ...] = np_img
            except OSError as e:
                self.read_errors.add(str(e) + '\n')
                continue
        print('THREAD {} HAS FINISHED'.format(self.set_type))


def images_in_paths(folder_path: str) -> List[str]:
    """
    Collects all images from one folder and return a list of paths
    :param folder_path:
    :return:
    """
    paths = []
    folder_path = os.path.join(os.getcwd(), folder_path)
    # extensions = set([])
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # extensions.add(os.path.splitext(file)[1])
            paths.append(os.path.join(root, file))
    # with open('extensions.txt', 'w') as f:
    #     f.writelines([extension + '\n' for extension in extensions])
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


def generate_h5py(file_list: List, img_size=256, hdf5_file_name: str = 'data', train_dim: float = 0.70, val_dim: float = 0.25, folder: str='h5_files'):
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

    # create dataset parts: X_<set_type> will contain the images, the Y_<set_type> the labels,
    # according to the labels of labels_dict
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(os.getcwd(), folder, hdf5_file_name), mode='w') as hdf5_out:
        hdf5_out.create_dataset('train_img', (len(file_dict['train']), img_size, img_size, 3), np.uint8)
        hdf5_out.create_dataset('val_img', (len(file_dict['val']), img_size, img_size, 3), np.uint8)
        hdf5_out.create_dataset('test_img', (len(file_dict['test']), img_size, img_size, 3), np.uint8)
        hdf5_out.create_dataset('train_mean', (img_size, img_size, 3), np.float32)
        hdf5_out.create_dataset('train_std', (img_size, img_size, 3), np.float32)
        hdf5_out.create_dataset('train_dim', (), np.int32, data=int(n*train_dim))
        hdf5_out.create_dataset('val_dim', (), np.int32, data=int(n*val_dim))
        hdf5_out.create_dataset('test_dim', (), np.int32, data=int(n*(1-train_dim-val_dim)))
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
            if thread.read_errors:
                with open('errors{}.txt'.format(i), 'w') as f:
                    f.writelines(thread.read_errors)
            if thread.training_variance is not None:
                # calculate the std using the variace array only for train set
                training_std = np.sqrt(thread.training_variance / (len(file_dict['train']) - 1))
                hdf5_out['train_mean'][...] = thread.training_mean_new
                hdf5_out['train_std'][...] = training_std


if __name__ == '__main__':
    output_path = os.path.join(os.getcwd(), 'resources')
    elements = int(1e5)  # number of images to keep
    res_path = os.path.join('E:\\dataset\\images_only')
    images_list = images_in_paths(os.path.join(res_path))
    random.shuffle(images_list)
    images_list = images_list[0:elements]
    generate_h5py(images_list, 256, 'ILSVRC_'+str(elements)+'.h5', folder=os.path.join(output_path, 'h5_files'))
