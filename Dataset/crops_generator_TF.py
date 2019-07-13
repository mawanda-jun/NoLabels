import numpy as np
import h5py
import random
import tensorflow as tf
from config import conf
import os
from PIL import Image
from typing import List

AUTOTUNE = tf.data.experimental.AUTOTUNE


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


class ImageGenerator:
    """
    ImageGenerator takes care to serve all path/to/images in the right way and to serve the dataset info about mean,
    standard deviation and the number of items in folder.
    """

    def __init__(self, data_path):
        """
        Open file once. The object ImageGenerator is kept alive for all the training (and evaluation). So lists and file
        are not loaded to disk yet, waiting for call function.
        :param data_path: path/to/data folder, in which there are train, val, test folders with images and the file 'info.h5' with
        std/mean and number of items in dataset
        """
        self.data_path = data_path
        self.h5f = h5py.File(os.path.join(data_path, 'info.h5'),
                             'r')  # it will be closed when the context will be terminated

    def __call__(self, dataset_type, num_classes, *args, **kwargs):
        """
        Instance is called with different dataset_type
        :param dataset_type:
        :param args:
        :param kwargs:
        :return:
        """
        paths = images_in_paths(os.path.join(self.data_path, dataset_type))
        labels = [random.randrange(num_classes) for _ in paths]
        return paths, labels


class CropsGenerator:
    """
    CropsGenerator takes care to load images from disk and convert, crop and serve them as a tf.data.Dataset
    """

    def __init__(self, conf, max_hamming_set):
        self.data_path = conf.data_path  # path to dataset folder
        self.img_generator = ImageGenerator(self.data_path)  # generates the instance to dataset files
        self.numChannels = conf.numChannels  # num of input image channels
        self.numCrops = conf.numCrops  # num of jigsaw crops
        self.original_dim = conf.original_dim  # size of the input image
        self.cropSize = conf.cropSize  # size of crop (255)
        self.cellSize = conf.cellSize  # size of each cell (75)
        self.tileSize = conf.tileSize  # size of tile in cell (64)
        self.colorJitter = conf.colorJitter  # number of pixels for color jittering
        self.batchSize = conf.batchSize  # training_batch_size
        self.val_batch_size = conf.val_batch_size  # validation batch size
        self.meanTensor, self.stdTensor = self.get_stats()  # get stats from dataset info
        # the <max_hamming_set> comes from file generated by <generate_hamming_set>. For further information see it.
        self.maxHammingSet = np.array(max_hamming_set, dtype=np.uint8)
        self.numClasses = conf.hammingSetSize  # number of different jigsaw classes

        # do not retrieve info about dataset with h5f['train_img'][:].shape since it loads the whole dataset into RAM
        self.num_train_batch = self.img_generator.h5f['train_dim'][...].astype(np.int32) // self.batchSize
        self.num_val_batch = self.img_generator.h5f['val_dim'][...].astype(np.int32) // conf.val_batch_size
        self.num_test_batch = self.img_generator.h5f['test_dim'][...].astype(np.int32) // self.batchSize

    def get_stats(self):
        """
        Return mean and std from dataset. It has been memorized. If not mean or std have been saved a KeyError is raised.
        :return:
        """
        mean = self.img_generator.h5f['train_mean'][:].astype(np.float32)
        std = self.img_generator.h5f['train_std'][:].astype(np.float32)
        if self.numChannels == 1:
            mean = np.expand_dims(mean, axis=-1)
            std = np.expand_dims(std, axis=-1)
        return mean, std

    def one_crop(self, hm_index, crop_x, crop_y, x):
        """
        This function creates one cropping at a time.
        It's sort of the contrary wrt previous behaviour. Now we find the hm_index crop we want to locate and we create
        it. As the result is stacked in the first axis, the order is kept as the label requires.
        :param hm_index: permutation index in the hamming set
        :param crop_x: x position for the 225 initial crop of the image
        :param crop_y: x position for the 225 initial crop of the image
        :param x: original image
        :return: one crop
        """

        # Define behaviour of col and rows to keep compatibility with previous code
        col = tf.math.mod(hm_index, 3)
        row = tf.math.divide(hm_index, 3)
        # cast to get rid of decimal values. However multiply takes only float32, so we have to re-cast it
        row = tf.cast(row, tf.int16)
        row = tf.cast(row, tf.float32)

        # create tf.constant to keep compatibility
        crop_x = tf.constant(crop_x, dtype=tf.float32)
        crop_y = tf.constant(crop_y, dtype=tf.float32)
        # random_x = tf.constant(random.randrange(self.cellSize - self.tileSize), dtype=tf.float32)
        random_x = float(random.randrange(self.cellSize - self.tileSize))
        # random_y = tf.constant(random.randrange(self.cellSize - self.tileSize), dtype=tf.float32)
        random_y = float(random.randrange(self.cellSize - self.tileSize))
        cell_size = tf.constant(self.cellSize, dtype=tf.float32)
        tile_size = tf.constant(self.tileSize, dtype=tf.float32)

        # partial operations
        col_cellsize = tf.math.multiply(col, cell_size)
        row_cellsize = tf.math.multiply(row, cell_size)
        x_start = tf.add(tf.math.add(crop_x, col_cellsize), random_x)
        y_start = tf.add(tf.math.add(crop_y, row_cellsize), random_y)
        x_next = tf.math.add(x_start, tile_size)
        y_next = tf.math.add(y_start, tile_size)

        # cast every value to int so we can take slices of x
        x_start = tf.cast(x_start, dtype=tf.int32)
        y_start = tf.cast(y_start, dtype=tf.int32)
        x_next = tf.cast(x_next, dtype=tf.int32)
        y_next = tf.cast(y_next, dtype=tf.int32)
        crop = x[y_start:y_next, x_start:x_next, :]

        # spatial jittering of crop
        crop = self.color_channel_jitter(crop)

        # ensure that resulting shape is correct
        tf.ensure_shape(crop, shape=(64, 64, 3))
        return crop

    def create_croppings(self, x: tf.Tensor, y: tf.Tensor, hamming_set: tf.Tensor):
        """
        Makes croppings from image
        The 3x3 grid is numbered as follows:
        0    1    2
        3    4    5
        6    7    8
        :param x:
        :param y:
        :param hamming_set:
        :return:
        """
        # retrieve shape of image
        y_dim, x_dim = x.shape[:2]

        # Have the x & y coordinate of the crop
        if x_dim != self.cropSize:
            # dimension of x is bigger than cropSize so we can take a square window inside image
            crop_x = random.randrange(x_dim - self.cropSize)
            crop_y = random.randrange(y_dim - self.cropSize)
        else:
            crop_x, crop_y = 0, 0
        # define variable before mapping

        # create lambda function for mapping
        one_crop_func = lambda hm_index: self.one_crop(hm_index, crop_x, crop_y, x)
        # this mapping takes one element at a time from <hamming_set> and serve it to croppings_func. So for
        # croppings_func hm_index is served from hamming_set, the other parameters from <create_croppings> function body
        # This map returns a tensor that has the one_crop stacked together in the first dimension
        croppings = tf.map_fn(one_crop_func, hamming_set)
        # change order of axis (move one_crop dimension from first to last)
        x = tf.transpose(croppings, [1, 2, 3, 0])
        return x, tf.one_hot(y, self.numClasses)

    def normalize_image(self, x: tf.Tensor, perm_index: tf.Tensor, hamming_set):
        """
        Normalize data one image at a time
        :param x: is a single images.
        :param perm_index:
        :param hamming_set:
        :return: image, normalized wrt dataset mean and std
        """
        # make it greyscale with probability 0.3%
        if random.random() < 0.3:
            x = 0.21 * x[..., 2] + 0.72 * x[..., 1] + 0.07 * x[..., 0]
            # expanding dimension to preserve net layout
            x = tf.expand_dims(x, axis=-1)
            x = tf.concat([x, x, x], axis=-1)

        # make the image distant from std deviation of the dataset
        x = tf.math.subtract(x, self.meanTensor)
        x = tf.math.divide(x, self.stdTensor)

        return x, perm_index, hamming_set

    def color_channel_jitter(self, img):
        """
        Spatial image jitter, aka movement of color channel in various manners
        """
        r_jit = random.randrange(-self.colorJitter, self.colorJitter)
        g_jit = random.randrange(-self.colorJitter, self.colorJitter)
        b_jit = random.randrange(-self.colorJitter, self.colorJitter)
        return tf.stack((
            tf.roll(img[:, :, 0], r_jit, axis=0),
            tf.roll(img[:, :, 1], g_jit, axis=1),
            tf.roll(img[:, :, 2], b_jit, axis=0)
        ), axis=2)

    def parse_path(self, path: tf.Tensor, label: tf.Tensor) -> (tf.Tensor, tf.Tensor, tf.Tensor):
        """
        Read image from disk and apply a label to it
        :param path: path to one image. This is a tf.Tensor and contains a string
        :return:
        """
        # read image from disk
        img = tf.io.read_file(path)
        # decode it as jpeg
        img = tf.image.decode_jpeg(img, channels=3)
        # cast to tensor with type tf.float32
        img = tf.cast(img, dtype=tf.float32)


        # perm_index = int(random.randrange(self.numClasses))
        # hamming_set = tf.constant(self.maxHammingSet[perm_index], dtype=tf.float32)
        # hamming_set = np.array(self.maxHammingSet[perm_index], dtype=np.float32)
        hamming_set = tf.cast(tf.gather(self.maxHammingSet, label), dtype=tf.float32)
        # perm_index = tf.constant(perm_index, dtype=tf.int32)
        return img, label, hamming_set

    def generate(self, mode='train'):
        """
        Generates the actual dataset. It uses all the functions defined above to read images from disk and create croppings.
        :param mode: train-val-test
        :return: tf.data.Dataset
        """
        parse_path_func = lambda x, y: self.parse_path(x, y)
        normalize_func = lambda x, y, z: self.normalize_image(x, y, z)
        create_croppings_func = lambda x, y, z: self.create_croppings(x, y, z)

        if mode == 'val':
            batch_size = self.val_batch_size
            n_el = self.num_val_batch
        else:
            batch_size = self.batchSize
            n_el = self.num_train_batch

        dataset = (tf.data.Dataset.from_tensor_slices(self.img_generator(mode, self.numClasses))
                   .shuffle(buffer_size=n_el * batch_size)
                   .map(parse_path_func, num_parallel_calls=AUTOTUNE)
                   .map(normalize_func, num_parallel_calls=AUTOTUNE)  # normalize input for mean and std
                   .map(create_croppings_func, num_parallel_calls=AUTOTUNE)  # create actual one_crop
                   .batch(batch_size)  # defined batch_size
                   .prefetch(AUTOTUNE)  # number of batches to be prefetch.
                   .repeat()  # repeats the dataset when it is finished
                   )
        return dataset


# UNCOMMENT ADDITION AND DIVISION PER MEAN AND STD BEFORE TRY TO SEE IMAGES
if __name__ == '__main__':
    os.chdir(os.pardir)
    with h5py.File(os.path.join('Dataset', conf.resources, conf.hammingFileName + str(conf.hammingSetSize) + '.h5'),
                   'r') as h5f:
        HammingSet = np.array(h5f['max_hamming_set'])

    data_reader = CropsGenerator(conf, HammingSet)

    iter = data_reader.generate(mode='train').make_initializable_iterator()
    x, labels = iter.get_next()

    with tf.Session() as sess:
        sess.run(iter.initializer)
        # returns a batch of images
        tiles, labels = sess.run([x, labels])
        # select only one (choose which in [0, batchSize)
        n_image = 4
        image = np.array(tiles[n_image], dtype=np.float32)
        first_label = np.array(labels[n_image])
        # from one_hot to number
        lbl = np.where(first_label == np.amax(first_label))[0][0]

        # create complete image with pieces (if label is correct then also will be image)
        complete = np.zeros((192, 192, 3))
        tile_size = data_reader.tileSize
        for i, v in enumerate(data_reader.maxHammingSet[lbl]):
            row = int(v / 3)
            col = v % 3
            y_start = row * tile_size
            x_start = col * tile_size
            complete[y_start:y_start + tile_size, x_start:x_start + tile_size] = image[:, :, :, i]

        Image.fromarray(np.array(complete, dtype=np.uint8)).show()
