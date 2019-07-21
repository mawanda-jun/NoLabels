"""
This file is the dataset generator for the target task.
It uses the structure of train/val/test set images created in file create_dataset_TL.py
It reads the info about the dataset from its info.h5 file, so it is important to load it before training.
"""
import os
import tensorflow as tf
import h5py
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataGenerator:
    def __init__(self, conf: any):
        self.conf = conf
        # dict containing 'train', 'validation' and 'test' path + labels
        self.data = {}
        self.data_path = conf.data_path  # path/to/train_val_test images
        # retrieve mean and std of train set
        with h5py.File(os.path.join(self.data_path, 'info.h5'), 'r') as f:
            self.mean = f['train_mean'][:].astype(np.float32)
            self.std = f['train_std'][:].astype(np.float32)

        # open file with listed classes
        with open(os.path.join(self.data_path, 'classes_with_sparse_labels.txt')) as f:
            self.classes = f.readlines()
        # generate classes dict so it can be possible to retrieve the name of the sparse label
        temp_dict = {}
        for c in self.classes:
            c = c.replace('\n', '')
            cl, sparse_label = c.split(' ')
            temp_dict[cl] = int(sparse_label)
        self.classes_dict = temp_dict

        # get paths from train, validation and test data
        self.training_data = os.path.join(self.data_path, 'train')
        self.validation_data = os.path.join(self.data_path, 'val')
        self.test_data = os.path.join(self.data_path, 'test')
        self.train_size = 0
        self.val_size = 0
        self.test_size = 0

        # prepare dataset paths
        self.create()

    @staticmethod
    def __parse_img(image: str, label: str):
        img = tf.io.read_file(image)
        img = tf.image.decode_jpeg(img, channels=3)
        # img = tf.image.central_crop(img, central_fraction=1)
        # img = tf.image.resize(img, [self.conf.cropSize, self.conf.cropSize]

        return img, label

    @staticmethod
    def __augment_data(img: str, label: str):
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_contrast(img, 4, 10)
        return img, label

    def __normalize_img(self, img: str, label: str):
        img = tf.cast(img, tf.float32)
        img = tf.math.subtract(img, self.mean)
        img = tf.math.divide(img, self.std)
        return img, label

    def __fetch_filenames_label(self, get_train: bool = False, get_test: bool = False, get_validation: bool = False):
        if get_train:
            train_data = self.data['train']
            return list(zip(*train_data))[0], list(zip(*train_data))[1]
        elif get_validation:
            validation_data = self.data['val']
            return list(zip(*validation_data))[0], list(zip(*validation_data))[1]
        elif get_test:
            test_data = self.data['test']
            return list(zip(*test_data))[0], list(zip(*test_data))[1]

    def create(self):
        train_labels = []
        for file in os.listdir(self.training_data):
            filename = os.path.basename(file)
            train_labels.append(self.classes_dict[filename.split('%')[0]])

        val_labels = []
        for file in os.listdir(self.validation_data):
            filename = os.path.basename(file)
            val_labels.append(self.classes_dict[filename.split('%')[0]])

        test_labels = []
        for file in os.listdir(self.test_data):
            filename = os.path.basename(file)
            test_labels.append(self.classes_dict[filename.split('%')[0]])

        train_paths = [os.path.join(self.training_data, path) for path in os.listdir(self.training_data)]
        val_paths = [os.path.join(self.validation_data, path) for path in os.listdir(self.validation_data)]
        test_paths = [os.path.join(self.test_data, path) for path in os.listdir(self.test_data)]
        train_set = list(zip(train_paths, train_labels))
        val_set = list(zip(val_paths, val_labels))
        test_set = list(zip(test_paths, test_labels))

        self.data['train'] = train_set
        self.data['val'] = val_set
        self.data['test'] = test_set

    def get_training_set(self):
        parse = lambda f, l: self.__parse_img(f, l)
        normalize = lambda f, l: self.__normalize_img(f, l)
        augment = lambda f, l: self.__augment_data(f, l)
        filenames, labels = self.__fetch_filenames_label(get_train=True)
        assert len(filenames) == len(labels), "filenames and labels should have the same lenght"
        # convert tuples into lists since from_tensor_slices does not recognize tuples
        filenames = [*filenames]
        labels = [*labels]
        self.train_size = len(filenames)

        return (tf.data.Dataset.from_tensor_slices((filenames, labels))
                .shuffle(buffer_size=self.train_size, reshuffle_each_iteration=True)
                # Perfect shuffling requires size of whole dataset
                .map(parse, num_parallel_calls=AUTOTUNE)
                .map(augment, num_parallel_calls=AUTOTUNE)
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .batch(self.conf.batchSize)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_validation_set(self):
        parse = lambda f, l: self.__parse_img(f, l)
        normalize = lambda f, l: self.__normalize_img(f, l)
        filenames, labels = self.__fetch_filenames_label(get_validation=True)
        assert len(filenames) == len(labels), "Filenames and labels should have same length"
        # convert tuples into lists since from_tensor_slices does not recognize tuples
        filenames = [*filenames]
        labels = [*labels]
        self.val_size = len(filenames)

        return (tf.data.Dataset.from_tensor_slices((filenames, labels))
                .shuffle(buffer_size=self.val_size, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .batch(self.conf.val_batch_size)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_test_set(self):
        parse = lambda f, l: self.__parse_img(f, l)
        normalize = lambda f, l: self.__normalize_img(f, l)
        filenames, labels = self.__fetch_filenames_label(get_validation=True)
        assert len(filenames) == len(labels), "Filenames and labels should have same length"
        # convert tuples into lists since from_tensor_slices does not recognize tuples
        filenames = [*filenames]
        labels = [*labels]
        self.test_size = len(filenames)

        return (tf.data.Dataset.from_tensor_slices((filenames, labels))
                .map(parse, num_parallel_calls=AUTOTUNE)
                .map(normalize, num_parallel_calls=AUTOTUNE)
                .batch(self.conf.test_batch_size)
                .repeat()
                .prefetch(AUTOTUNE))


# UNCOMMENT ADDITION AND DIVISION PER MEAN AND STD BEFORE TRY TO SEE IMAGES
if __name__ == '__main__':
    from config_ft import conf_ft as conf
    from PIL import Image

    os.chdir(os.pardir)

    data_reader = DataGenerator(conf)

    iter = data_reader.get_training_set().make_initializable_iterator()
    x, labels = iter.get_next()

    with tf.Session() as sess:
        sess.run(iter.initializer)
        # returns a batch of images
        images, labels = sess.run([x, labels])
        # select only one (choose which in [0, batchSize)
        n_image = 4
        image = np.array(images[n_image], dtype=np.float32)
        label = np.array(labels[n_image])

        Image.fromarray(np.array(image, dtype=np.uint8)).show()

