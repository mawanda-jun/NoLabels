from os import listdir
import os
import random
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class DataGenerator:
    def __init__(self, conf):
        self.conf = conf
        self.data = None

    def __parse_img(self, image: str, label: str):
        img = tf.io.read_file(image)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)
        img = tf.image.central_crop(img, central_fraction=1)
        img = tf.image.resize(img, [self.conf.cropSize, self.conf.cropSize])
        return img, tf.one_hot(label, 100)

    def __fetch_filesnames_label(self, get_train: bool = False, get_test: bool = False, get_validation: bool = False):
        n_examples = len(self.data)

        idx_train = int(n_examples * self.conf.train_size)
        idx_validation = int(n_examples * self.conf.val_size) + idx_train
        idx_test = int(n_examples * self.conf.test_size) + idx_validation

        if get_train:
            train_data = self.data[0:idx_train]
            return list(zip(*train_data))[0], list(zip(*train_data))[1]
        elif get_validation:
            validation_data = self.data[idx_train:idx_validation]
            return list(zip(*validation_data))[0], list(zip(*validation_data))[1]
        elif get_test:
            test_data = self.data[idx_validation:idx_test]
            return list(zip(*test_data))[0], list(zip(*test_data))[1]

    def create(self):
        path = os.path.join(self.conf.data_path)
        classes = listdir(path)
        image_path = []
        image_class = []

        for i in range(0, 100):
            clas_path = os.path.join(self.conf.data_path, classes[i])

            for filename in listdir(clas_path):
                image_path.append(os.path.join(self.conf.data_path, classes[i], filename))
                image_class.append(i)

        a = list(zip(image_path, image_class))

        self.data = a

    def shuffle(self, seed: int = int(random.random() * 100)):
        random.Random(seed).shuffle(self.data)

    def get_data(self):
        return self.data

    def get_training_set(self):
        parse = lambda f, l: self.__parse_img(f, l)
        filenames, labels = self.__fetch_filesnames_label(get_train=True)

        num_samples = len(filenames)

        return (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
                # Perfect shuffling requires size of whole dataset
                .map(parse, num_parallel_calls=AUTOTUNE)
                .batch(self.conf.batchSize)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_validation_set(self):
        parse = lambda f, l: self.__parse_img(f, l)
        filenames, labels = self.__fetch_filenames_labels(get_validation=True)
        assert len(filenames) == len(labels), "Filenames and labels should have same length"

        num_samples = len(filenames)
        return (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                .batch(self.conf.val_batch_size)
                .repeat()
                .prefetch(AUTOTUNE))

    def get_validation_set(self):
        parse = lambda f, l: self.__parse_img(f, l)
        filenames, labels = self.__fetch_filesnames_label(get_validation=True)
        assert len(filenames) == len(labels), "Filenames and labels should have same length"

        num_samples = len(filenames)
        return (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
                .shuffle(buffer_size=num_samples, reshuffle_each_iteration=True)
                .map(parse, num_parallel_calls=AUTOTUNE)
                .batch(self.conf.test_batch_size)
                .repeat()
                .prefetch(AUTOTUNE))

