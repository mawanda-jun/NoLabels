import numpy as np
import h5py
import random
import tensorflow as tf


class H5Generator:
    def __init__(self, file):
        self.file = file

    def __call__(self, mode, *args, **kwargs):
        with h5py.File(self.file, 'r') as hf:
            for im in hf[mode]:
                yield im


class CropsGenerator:
    """
    Read HD5F file, one batch at a time, and serve a jigsaw puzzle.
    """
    def __init__(self, conf, max_hamming_set):
        self.data_path = conf.data_path  # path to hd5f file
        self.numChannels = conf.numChannels  # num of input image channels
        self.numCrops = conf.numCrops  # num of jigsaw crops
        self.cropSize = conf.cropSize  # size of crop (255)
        self.cellSize = conf.cellSize  # size of each cell (75)
        self.tileSize = conf.tileSize  # size of tile in cell (64)
        self.colorJitter = conf.colorJitter  # number of pixels for color jittering
        self.batchSize = conf.batchSize  # training_batch_size
        self.meanTensor, self.stdTensor = self.get_stats()
        # the <max_hamming_set> comes from file generated by <generate_hamming_set>. For further information see it.
        self.maxHammingSet = np.array(max_hamming_set, dtype=np.uint8)
        self.numClasses = self.maxHammingSet.shape[0]  # number of different jigsaw classes

        with h5py.File(self.data_path, 'r') as h5f:
            # TODO: cambiare X_train in train_img e tutti gli altri. Da cambiare anche in AlexNet\Siamese
            self.num_train = h5f['X_train'][:].shape[0]
            self.num_train_batch = self.num_train // self.batchSize
            self.num_val = h5f['X_val'][:].shape[0]
            self.num_val_batch = self.num_val // self.batchSize
            self.num_test = h5f['X_test'][:].shape[0]
            self.num_test_batch = self.num_test // self.batchSize

        self.batchIndexTrain = 0
        self.batchIndexVal = 0
        self.batchIndexTest = 0

    def get_stats(self):
        """
        Return mean and std from dataset. It has been memorized. If not mean or std have been saved a KeyError is raised.
        :return:
        """
        with h5py.File(self.data_path, 'r') as h5f:
            mean = h5f['train_mean'][:].astype(np.float32)
            std = h5f['train_std'][:].astype(np.float32)
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
        # final_crops = tf.Variable(final_crops)
        crops_per_side = int(np.sqrt(self.numCrops))  # 9 crops -> 3 crops per side
        # x_cuts, y_cuts = [], []
        for row in range(crops_per_side):
            for col in range(crops_per_side):
                x_start = crop_x + col * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                y_start = crop_y + row * self.cellSize + random.randrange(self.cellSize - self.tileSize)
                final_crops[:, :, :, self.maxHammingSet[perm_index, row * crops_per_side + col]] = \
                    image[y_start:y_start + self.tileSize, x_start:x_start + self.tileSize, :]
        #         PROBABILMENTE NON FUNZIONA PERCHE' SI CERCA DI TAGLIARE UN TENSORE E DI ASSEGNARLO AD UN NUMPY ARRAY
        #         x_cuts.append([x_start, row, col])
        #
        #         y_cuts.append([y_start, row, col])
        #         # Put the crop in the list of pieces randomly according to the number picked from max_hamming_set
        #
        # img_stacked = []
        # for i in range(self.numCrops):
        #     row = x_cuts[1]
        #     col = x_cuts[2]
        #     img_stacked.append({'image': tf.image.crop_to_bounding_box(image, x_cuts[i][0], y_cuts[i][0], self.tileSize, self.tileSize),
        #                         'perm': self.maxHammingSet[perm_index, row * crops_per_side + col]}
        #                        )
        # img_stacked = sorted(img_stacked, key=lambda x: x['perm'])
        # final_crops = tf.stack(img_stacked, axis=0)

        return final_crops, perm_index

    def __batch_generation_normalized(self, x: np.array):
        """
        Normalize data one image at a time and produce a batch of cropped image
        :param x: is a list of images. |x| == <self.batchSize>
        :return:
        """
        # modify each image individually
        x -= self.meanTensor
        x /= self.stdTensor
        # defining "labels" for every image of batch
        y = np.empty(self.batchSize)
        # create <numCrops> long array for every tile of one image
        tile = np.empty((self.batchSize, self.tileSize, self.tileSize, self.numChannels), np.float32)
        X = [tile for _ in range(self.numCrops)]

        # create array with random indexes form max_hamming_set. We do this way to have always different permutations
        # per batch
        random_permutations_indexes = np.zeros(self.batchSize)
        random_permutations_indexes[0] = random.randrange(self.numClasses)
        i = 1
        while i < self.batchSize:
            tmp = random.randrange(self.numClasses)
            if tmp not in random_permutations_indexes:
                random_permutations_indexes[i] = tmp
                i += 1

        assert len(random_permutations_indexes) == self.batchSize

        for num_image in range(self.batchSize):
            # really transform each image in its crops
            perm_index = int(random_permutations_indexes[num_image])
            # y[num_image] will be equal to perm_index. We keep this way
            tiles, y[num_image] = self.create_croppings(x[num_image], perm_index)
            for position_in_crop in range(self.numCrops):
                X[position_in_crop][num_image, :, :, :] = tiles[:, :, :, position_in_crop]

        return X, y

    def __image_generation_normalized(self, x: tf.Tensor, perm_index: int):
        """
        Normalize data and produce whole dataset cropped
        :param x: is an image
        :param perm_index: index of permutation
        :return:
        """
        # modify each image individually
        x = tf.subtract(x, self.meanTensor)
        x = tf.divide(x, self.stdTensor)
        # defining "labels" for every image of batch
        # y = np.empty(self.num_train)
        # create <numCrops> long array for every tile of one image
        tile = np.empty((self.tileSize, self.tileSize, self.numChannels), np.float32)
        X = np.array([tile for _ in range(self.numCrops)])
        X = tf.convert_to_tensor(X, tf.float32)

        # create array with random indexes form max_hamming_set. We do this way to have always different permutations
        # per batch
        # random_permutations_indexes = np.zeros(self.batchSize)
        # random_permutations_indexes[0] = random.randrange(self.numClasses)
        # i = 1
        # while i < self.batchSize:
        #     tmp = random.randrange(self.numClasses)
        #     if tmp not in random_permutations_indexes:
        #         random_permutations_indexes[i] = tmp
        #         i += 1
        #
        # assert len(random_permutations_indexes) == self.batchSize

        tiles, y = self.create_croppings(x, perm_index)

        for position_in_crop in range(self.numCrops):
            X[position_in_crop][None, :, :, :] = tiles[position_in_crop]
        # for num_image in range(self.batchSize):
            # really transform each image in its crops
            # perm_index = int(random_permutations_indexes[num_image])
            # y[num_image] will be equal to perm_index. We keep this way

        return X, self.one_hot(self.maxHammingSet[y])

    def one_hot(self, y):
        """
        OneHot encoding for labels in y.
        :param y: labels
        :return: y in the format of OneHot
        """
        return np.array([[1 if y[i] == j else 0 for j in range(self.numClasses)] for i in range(y.shape[0])])

    def generate(self, mode='train'):
        """
        Generates batch and serve always new one
        :param mode:
        :return:
        """
        h5f_label = None
        batch_index = -2
        # TODO: cambiare X_train in train_img e tutti gli altri
        if mode == 'train':
            h5f_label = 'X_train'
            batch_index = self.batchIndexTrain
            self.batchIndexTrain += 1
            if self.batchIndexTrain == self.num_train_batch:
                self.batchIndexTrain = 0
        elif mode == 'val':
            h5f_label = 'X_val'
            batch_index = self.batchIndexVal
            self.batchIndexVal += 1
            if self.batchIndexVal == self.num_val_batch:
                self.batchIndexVal = 0
        elif mode == 'test':
            h5f_label = 'X_test'
            batch_index = self.batchIndexTest
            self.batchIndexTest += 1
            if self.batchIndexTest == self.num_test_batch:
                self.batchIndexTest = 0

        with h5py.File(self.data_path, 'r') as h5f:
            x = h5f[h5f_label][batch_index * self.batchSize:(batch_index + 1) * self.batchSize, ...]
        if self.numChannels == 1:
            x = np.expand_dims(x, axis=-1)
        X, y = self.__batch_generation_normalized(x.astype(np.float32))
        return np.transpose(np.array(X), axes=[1, 2, 3, 4, 0]), self.one_hot(y)

    def generateTF(self, mode='train'):
        """
        Generates batch and serve always new one
        :param mode:
        :return:
        """
        h5f_label = None
        # batch_index = -2
        # TODO: cambiare X_train in train_img e tutti gli altri
        if mode == 'train':
            h5f_label = 'X_train'
            # batch_index = self.batchIndexTrain
            # self.batchIndexTrain += 1
            # if self.batchIndexTrain == self.num_train_batch:
            #     self.batchIndexTrain = 0
        elif mode == 'val':
            h5f_label = 'X_val'
            # batch_index = self.batchIndexVal
            # self.batchIndexVal += 1
            # if self.batchIndexVal == self.num_val_batch:
            #     self.batchIndexVal = 0
        elif mode == 'test':
            h5f_label = 'X_test'
            # batch_index = self.batchIndexTest
            # self.batchIndexTest += 1
            # if self.batchIndexTest == self.num_test_batch:
            #     self.batchIndexTest = 0
        parse_fn = lambda img: self.__image_generation_normalized(img, random.randrange(self.numClasses))
        dataset = (tf.data.Dataset.from_generator(H5Generator(h5f_label), tf.float32, tf.TensorShape([256, 256, 3]))
                   .map(parse_fn)
                   .batch(self.batchSize)
                   .prefetch(1)
                   )
        iterator = dataset.make_initializable_iterator()
        
        return iterator
        # with h5py.File(self.data_path, 'r') as h5f:
        #     x = h5f[h5f_label][batch_index * self.batchSize:(batch_index + 1) * self.batchSize, ...]
        # if self.numChannels == 1:
        #     x = np.expand_dims(x, axis=-1)
        # X, y = self.__whole_generation_normalized(x.astype(np.float32), mode)
        # return np.transpose(np.array(X), axes=[1, 2, 3, 4, 0]), self.one_hot(y)


    def randomize(self):
        """ Randomizes the order of data samples"""
        with h5py.File(self.data_path, 'a') as h5f:
            train_img = h5f['X_train'][:].astype(np.float32)
            permutation = np.random.permutation(train_img.shape[0])
            train_img = train_img[permutation, :, :, :]
            del h5f['X_train']
            h5f.create_dataset('X_train', data=train_img)

    def color_channel_jitter(self, image):
        """
        Explain
        """
        # Determine the dimensions of the array, minus the crop around the border
        # of 4 pixels (threshold margin due to 2 pixel jitter)
        x_dim = image.shape[0] - self.colorJitter * 2
        y_dim = image.shape[1] - self.colorJitter * 2
        # Determine the jitters in all directions
        R_xjit = random.randrange(self.colorJitter * 2 + 1)
        R_yjit = random.randrange(self.colorJitter * 2 + 1)
        # Seperate the colour channels
        return_array = np.empty((x_dim, y_dim, 3), np.float32)
        for colour_channel in range(3):
            return_array[:, :, colour_channel] = \
                image[R_xjit:x_dim +R_xjit, R_yjit:y_dim + R_yjit, colour_channel]
        return return_array
