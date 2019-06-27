import numpy as np
import h5py
import random
import tensorflow as tf


class H5Generator:
    """
    This is a generator of the file. We don't have to open it every time: we just open it once and then call it
    with exact parameters. In addition I've created a "prefetch" function, so the function takes care to pre-load the
    dataset and keep the training more smooth.
    """

    def __init__(self, file):
        """
        Open file once. The object CropsGenerator is kept alive for all the training (and evaluation). So the file will
        remain opened for the whole time.
        :param file:
        """
        self.h5f = h5py.File(file, 'r')  # it will be closed when the context will be terminated

    def __call__(self, mode, *args, **kwargs):
        for img in self.h5f[mode]:
            yield img


class CropsGenerator:
    """
    Read HD5F file, one batch at a time, and serve a jigsaw puzzle.
    """
    def __init__(self, conf, max_hamming_set):
        self.data_path = conf.data_path  # path to hd5f file
        self.img_generator = H5Generator(self.data_path)
        self.numChannels = conf.numChannels  # num of input image channels
        self.numCrops = conf.numCrops  # num of jigsaw crops
        self.cropSize = conf.cropSize  # size of crop (255)
        self.cellSize = conf.cellSize  # size of each cell (75)
        self.tileSize = conf.tileSize  # size of tile in cell (64)
        self.colorJitter = conf.colorJitter  # number of pixels for color jittering
        self.batchSize = conf.batchSize  # training_batch_size
        self.val_batch_size = conf.val_batch_size
        self.conf = conf
        self.meanTensor, self.stdTensor = self.get_stats()
        # the <max_hamming_set> comes from file generated by <generate_hamming_set>. For further information see it.
        self.maxHammingSet = np.array(max_hamming_set, dtype=np.uint8)
        self.numClasses = conf.hammingSetSize  # number of different jigsaw classes

        # shapes of datasets (train, validation, test):
        # (140000, 256, 256, 3)
        # (49999, 256, 256, 3)
        # (9998, 256, 256, 3)
        self.num_train_batch = conf.N_train_imgs // self.batchSize
        self.num_val_batch = conf.N_val_imgs // self.batchSize
        self.num_test_batch = conf.N_test_imgs // self.batchSize

    def get_stats(self):
        """
        Return mean and std from dataset. It has been memorized. If not mean or std have been saved a KeyError is raised.
        :return:
        """
        # with h5py.File(self.data_path, 'r') as h5f:
        mean = self.img_generator.h5f['train_mean'][:].astype(np.float32)
        std = self.img_generator.h5f['train_std'][:].astype(np.float32)
        if self.numChannels == 1:
            mean = np.expand_dims(mean, axis=-1)
            std = np.expand_dims(std, axis=-1)
        return mean, std

    def create_croppings(self, image: np.array, perm_index: int):
        """
        Makes croppings from image
        The 3x3 grid is numbered as follows:
        0    1    2
        3    4    5
        6    7    8
        :param image = 3D numpy array
        :param perm_index = index of referred permutation in max_hamming_set
        :return array of croppings (<num_croppings>) made of (heigh x width x colour channels) arrays
        """
        # Jitter the colour channel
        # NOT IMPLEMENTED YET
        # image = self.color_channel_jitter(image)

        y_dim, x_dim = image.shape[:2]
        # Have the x & y coordinate of the crop
        if x_dim != self.cropSize:
            # dimension of image is bigger than cropSize aka it has not been cropped before
            crop_x = random.randrange(x_dim - self.cropSize)
            crop_y = random.randrange(y_dim - self.cropSize)
        else:
            crop_x, crop_y = 0, 0

        final_crops = np.zeros((self.tileSize, self.tileSize, self.numChannels, self.numCrops), dtype=np.float32)  # array with crops
        crops_per_side = int(np.sqrt(self.numCrops))  # 9 crops -> 3 crops per side
        for row in range(crops_per_side):
            for col in range(crops_per_side):
                x_start = crop_x + col * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                y_start = crop_y + row * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                # Put the crop in the list of pieces randomly according to the number picked from max_hamming_set
                crop = image[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize, :]
                crop = self.color_channel_jitter(crop)  # jitter every image in a random way
                final_crops[:, :, :, self.maxHammingSet[perm_index, row * crops_per_side + col]] = crop
                # final_crops[:, :, :, self.maxHammingSet[perm_index, row * crops_per_side + col]] = \
                #     image[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize, :]
        return final_crops, perm_index

    def __single_generation_normalized(self, x: np.array):
        """
        Normalize data one image at a time and produce a batch of cropped image
        :param x: is a single images.
        :return:
        """
        # make it greyscale with probability 0.3% as in the paper
        if random.random() < 0.3:
            b = x[..., 0]
            g = x[..., 1]
            r = x[..., 2]
            x = 0.21 * r + 0.72 * g + 0.07 * b
            # expanding dimension to preserve net layout
            x = np.expand_dims(x, axis=-1)
            x = np.concatenate([x, x, x], axis=-1)

        # make the image distant from std deviation of the dataset
        x -= self.meanTensor
        x /= self.stdTensor

        # create <numCrops> long array for every tile of one image
        tile = np.empty((self.tileSize, self.tileSize, self.numChannels), np.float32)
        X = [tile for _ in range(self.numCrops)]

        perm_index = int(random.randrange(self.numClasses))
        tiles, y = self.create_croppings(x, perm_index)
        for position_in_crop in range(self.numCrops):
            X[position_in_crop][:, :, :] = tiles[:, :, :, position_in_crop]

        return X, y

    def one_hot(self, y):
        """
        OneHot encoding for y label.
        :param y: label
        :return: y in the format of OneHot
        """
        tmp = np.zeros(self.numClasses)
        tmp[y] = 1
        # tmp = tf.one_hot(y, self.numClasses)
        return tmp
        # return np.array([1 if y == j else 0 for j in range(self.numClasses)])

    def yield_cropped_images(self, mode='train'):
        """
        Generates batch and serve always new one
        :param mode:
        :return:
        """
        h5f_label = mode + '_img'

        for x in self.img_generator(h5f_label):
            X, y = self.__single_generation_normalized(x.astype(np.float32))
            yield np.transpose(np.array(X), axes=[1, 2, 3, 0]), self.one_hot(y)

    def generate(self, mode='train'):
        norm_func = lambda x, y: norm(x, y)
        if mode == 'val':
            batch_size = self.val_batch_size
        else:
            batch_size = self.batchSize
        dataset = (tf.data.Dataset.from_generator(
            lambda: self.yield_cropped_images(mode),
            (tf.float32, tf.float32),
            (tf.TensorShape([self.tileSize, self.tileSize, self.numChannels, self.numCrops]),
             tf.TensorShape([self.conf.hammingSetSize])))
                   # .map(norm_func, num_parallel_calls=8)
                   .batch(batch_size)
                   # .shuffle(self.img_generator.h5f[mode+'_img'].shape[0])
                   .prefetch(1)
                   .repeat()
                   )
        return dataset

    def color_channel_jitter(self, img):
        """
        Spatial image jitter, aka movement of color channel in various manners
        """
        r_jit = random.randrange(-self.colorJitter, self.colorJitter)
        g_jit = random.randrange(-self.colorJitter, self.colorJitter)
        b_jit = random.randrange(-self.colorJitter, self.colorJitter)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        return np.dstack((
            np.roll(R, r_jit, axis=0),
            np.roll(G, g_jit, axis=1),
            np.roll(B, b_jit, axis=0)
        ))


def norm(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    # x = tf.to_float(x)
    x = tf.math.divide(
        tf.subtract(x, tf.reduce_min(x)),
        tf.subtract(tf.reduce_max(x), tf.reduce_min(x)))
    return x, y
