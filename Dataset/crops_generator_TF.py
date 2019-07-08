import numpy as np
import h5py
import random
import tensorflow as tf
from config import conf
import os
from PIL import Image
import asyncio

AUTOTUNE = tf.data.experimental.AUTOTUNE


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
        # Preload slice of dataset to get faster access to dataset. Test dataset is not preload on default.
        self.b_dim = 2000
        self.buffer = {
            'train0': self.h5f['train_img'][0 * self.b_dim:self.b_dim],
            'train1': self.h5f['train_img'][self.b_dim:2 * self.b_dim],
            'val0': self.h5f['val_img'][0 * self.b_dim:self.b_dim],
            'val1': self.h5f['val_img'][self.b_dim:self.b_dim:2 * self.b_dim],
            'test0': [],
            'test1': [],
            'train_index': 1,  # two buffers are already been loaded
            'train_max': self.h5f['train_dim'][...].astype(np.int32) // self.b_dim,
            'val_index': 1,  # two buffers are already been loaded
            'val_max': self.h5f['val_dim'][...].astype(np.int32) // self.b_dim,
            'test_index': 0,
            'test_max': self.h5f['test_dim'][...].astype(np.int32) // self.b_dim,
        }
        # we need to setup a new event loop to force execution of
        self.loop = asyncio.new_event_loop()
        # set task for both tasks
        self.task1 = None
        self.task2 = None

    def __call__(self, mode, num_classes, hamming_set, *args, **kwargs):
        # continue until tasks are finished. Repeat() should call the function again
        while self.buffer[mode + '_index'] < self.buffer[mode + '_max']:
            # this is the first round, so self.task2 would be empty
            if self.task2 is not None:
                # command to wait execution of self.task2. This would never be a big await
                self.loop.run_until_complete(self.task2)
            # load images form buffer 0
            for img in self.buffer[mode + '0']:
                perm_index = int(random.randrange(num_classes))
                yield img, perm_index, hamming_set[perm_index]

            # buffer0 is now finished: we increment its index, order to refill it with self.task1, then we go to the next
            # buffer
            self.buffer[mode + '_index'] += 1
            self.task1 = self.loop.create_task(self.fill_buffer(mode, '0'))
            # Quando ha finito la prima parte vuol dire che il buffer0 e' vuoto e si va sul buffer1 per non perdere prestazioni
            # nel frattempo si ricarica il buffer0
            for img in self.buffer[mode + '1']:
                perm_index = int(random.randrange(num_classes))
                yield img, perm_index, hamming_set[perm_index]
            self.buffer[mode + '_index'] += 1
            self.loop.run_until_complete(self.task1)
            self.task2 = self.loop.create_task(self.fill_buffer(mode, '1'))
        # when the dataset has been iterated wholly it starts again from 0
        self.buffer[mode + '_index'] = 0

    async def fill_buffer(self, mode, n_buffer):
        try:
            self.buffer[mode + n_buffer] = \
                self.h5f[mode + '_img'][
                self.buffer[mode + '_index'] * self.b_dim:(self.buffer[mode + '_index'] + 1) * self.b_dim]
        except IndexError:
            # non serve fare nulla: il while posto sopra da' gia' sicurezza di uscita dal ciclo qualora si cerchi di
            # caricare una porzione non accessibile.
            pass


class CropsGenerator:
    """
    Read HD5F file, one batch at a time, and serve a jigsaw puzzle.
    """

    def __init__(self, conf, max_hamming_set):
        self.data_path = conf.data_path  # path to hd5f file
        self.img_generator = H5Generator(self.data_path)
        self.numChannels = conf.numChannels  # num of input image channels
        self.numCrops = conf.numCrops  # num of jigsaw crops
        self.original_dim = conf.original_dim  # size of the input image
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
        self.numClasses = conf.hammingSetSize  # number of different jigsaw classes (5)

        # shapes of datasets (train, validation, test):
        # (70000, 256, 256, 3)
        # (25000, 256, 256, 3)
        # (5000, 256, 256, 3)
        # do not retrieve info about dataset with h5f['train_img'][:].shape since it loads the whole dataset into RAM
        self.num_train_batch = self.img_generator.h5f['train_dim'][...].astype(np.int32) // self.batchSize
        self.num_val_batch = self.img_generator.h5f['val_dim'][...].astype(np.int32) // self.batchSize
        self.num_test_batch = self.img_generator.h5f['test_dim'][...].astype(np.int32) // self.batchSize
        # self.num_train_batch = conf.N_train_imgs // self.batchSize
        # self.num_val_batch = conf.N_val_imgs // self.batchSize
        # self.num_test_batch = conf.N_test_imgs // self.batchSize

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

    def one_crop(self, hm_index, crop_x, crop_y, x):
        # It's sort of the contrary wrt previous behaviour. Now we find the hm_index crop we want to locate and
        # we create its cropping. Then we stack in order, so the hamming_set order is kept.

        # Define behaviour of col and rows to keep compatibility with previous code
        col = tf.math.mod(hm_index, 3)
        row = tf.math.divide(hm_index, 3)
        # cast to get rid of decimal values. However multiply takes only float32, so we have to re-cast it
        row = tf.cast(row, tf.int16)
        row = tf.cast(row, tf.float32)

        # create tf.constant to keep compatibility
        crop_x = tf.constant(crop_x, dtype=tf.float32)
        crop_y = tf.constant(crop_y, dtype=tf.float32)
        random_x = tf.constant(random.randrange(self.cellSize - self.tileSize), dtype=tf.float32)
        random_y = tf.constant(random.randrange(self.cellSize - self.tileSize), dtype=tf.float32)
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
        :param x = 3D numpy array
        :param perm_index = index of referred permutation in max_hamming_set
        :return array of croppings (<num_croppings>) made of (heigh x width x colour channels) arrays
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
        return x, y

    def normalize_image(self, x: tf.Tensor, y: tf.Tensor, z):
        """
        Normalize data one image at a time
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
            x = tf.expand_dims(x, axis=-1)
            x = tf.concat([x, x, x], axis=-1)

        # make the image distant from std deviation of the dataset
        x = tf.math.subtract(x, self.meanTensor)
        x = tf.math.divide(x, self.stdTensor)

        return x, y, z

    def one_hot(self, x, y):
        """
        OneHot encoding for y label.
        :param y: label
        :return: y in the format of OneHot
        """
        return x, tf.one_hot(y, self.numClasses)

    def color_channel_jitter(self, img):
        """
        Spatial image jitter, aka movement of color channel in various manners
        """
        if self.colorJitter == 0:
            return img
        r_jit = random.randrange(-self.colorJitter, self.colorJitter)
        g_jit = random.randrange(-self.colorJitter, self.colorJitter)
        b_jit = random.randrange(-self.colorJitter, self.colorJitter)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        return tf.stack((
            tf.roll(R, r_jit, axis=0),
            tf.roll(G, g_jit, axis=1),
            tf.roll(B, b_jit, axis=0)
        ), axis=2)

    def generate(self, mode='train'):
        normalize_func = lambda x, y, z: self.normalize_image(x, y, z)
        create_croppings_func = lambda x, y, z: self.create_croppings(x, y, z)
        onehot_func = lambda x, y: self.one_hot(x, y)

        if mode == 'val':
            batch_size = self.val_batch_size
        else:
            batch_size = self.batchSize

        # h5f_label = mode + '_img'
        dataset = (tf.data.Dataset.from_generator(
            lambda: self.img_generator(mode, self.numClasses, self.maxHammingSet),  # generator
            (tf.float32, tf.int32, tf.float32),  # input types
            (tf.TensorShape([self.original_dim, self.original_dim, self.numChannels]),  # shapes of input types
             tf.TensorShape(()),
             tf.TensorShape([self.numCrops])))
                   .map(normalize_func, num_parallel_calls=AUTOTUNE)  # normalize input for mean and std
                   .map(create_croppings_func, num_parallel_calls=AUTOTUNE)  # create actual one_crop
                   .map(onehot_func, num_parallel_calls=AUTOTUNE)  # convert label into one_hot encoding
                   .batch(batch_size)  # defined batch_size
                   .prefetch(AUTOTUNE)  # number of batches to be prefetch.
                   .repeat()  # repeats the dataset when it is finished
                   )
        return dataset


# UNCOMMENT ADDITION AND DIVISION PER MEAN AND STD BEFORE TRY TO SEE IMAGES
if __name__ == '__main__':
    os.chdir(os.pardir)
    with h5py.File(os.path.join('Dataset', conf.resources, conf.hammingFileName + str(conf.hammingSetSize) + '.h5'), 'r') as h5f:
        HammingSet = np.array(h5f['max_hamming_set'])

    data_reader = CropsGenerator(conf, HammingSet)

    iter = data_reader.generate(mode='train').make_initializable_iterator()
    x, labels = iter.get_next()

    with tf.Session() as sess:
        sess.run(iter.initializer)
        # returns a batch of images
        tiles, labels = sess.run([x, labels])
        # select only one (choose which in [0, batchSize)
        n_image = 10
        image = np.array(tiles[n_image], dtype=np.float32)
        first_label = np.array(labels[n_image])
        # from one_hot to number
        lbl = np.where(first_label == np.amax(first_label))[0][0]

        # create complete image with pieces (if label is correct then also will be image)
        complete = np.zeros((192, 192, 3))
        tile_size = data_reader.tileSize
        for i, v in enumerate(data_reader.maxHammingSet[lbl]):
            row = int(v/3)
            col = v % 3
            y_start = row*tile_size
            x_start = col*tile_size
            complete[y_start:y_start + tile_size, x_start:x_start + tile_size] = image[:, :, :, i]

        Image.fromarray(np.array(complete, dtype=np.uint8)).show()
