from time import time
import h5py
import numpy as np
from PIL import Image
import glob
import random
import os


def generate_data(files, img_size=256, TEST_SUBSET_DATA=True, hdf5_file_name='DATA'):
    """
    Generate and save train, validation and test data
    :param files: file names
    :param img_size: size (both width and height) of the output images
    :param TEST_SUBSET_DATA: True when only generating from a subset of all files
    :param hdf5_file_name: name of the output HDF5 file
    """
    training_mean_new = np.zeros((img_size, img_size, 3), dtype=np.float32)
    training_mean_old = np.zeros((img_size, img_size, 3), dtype=np.float32)
    training_variance = np.zeros((img_size, img_size, 3), dtype=np.float32)
    # Shuffle the data
    random.shuffle(files)
    # Split data 70% train, 15% validation, 15% test
    files_dict = {}
    # Create the HDF5 output file
    if TEST_SUBSET_DATA:
        # Create small list of files for training subset of data
        files_dict["train_img"] = files[:500]
        files_dict["val_img"] = files[500:750]
        files_dict["test_img"] = files[750:1000]
        print("Length of files array: {}, but only getting 1000 images!".format(len(files)))

        hdf5_output = h5py.File(hdf5_file_name, mode='w')
        hdf5_output.create_dataset("train_img", (len(files_dict["train_img"]), img_size, img_size, 3),
                                   np.uint8, compression="gzip")
        hdf5_output.create_dataset("val_img", (len(files_dict["val_img"]), img_size, img_size, 3),
                                   np.uint8, compression="gzip")
        hdf5_output.create_dataset("test_img", (len(files_dict["test_img"]), img_size, img_size, 3),
                                   np.uint8, compression="gzip")
        hdf5_output.create_dataset("train_mean", (img_size, img_size, 3), np.float32, compression="gzip")
        hdf5_output.create_dataset("train_std", (img_size, img_size, 3), np.float32, compression="gzip")

    else:
        files_dict["train_img"] = files[:int(0.7 * len(files))]
        files_dict["val_img"] = files[int(0.7 * len(files)):int(0.85 * len(files))]
        files_dict["test_img"] = files[int(0.85 * len(files)):]
        print("Length of files array: {}".format(len(files)))

        hdf5_output = h5py.File(hdf5_file_name, mode='w')
        hdf5_output.create_dataset("train_img", (len(files_dict["train_img"]), img_size, img_size, 3), np.uint8)
        hdf5_output.create_dataset("val_img", (len(files_dict["val_img"]), img_size, img_size, 3), np.uint8)
        hdf5_output.create_dataset("test_img", (len(files_dict["test_img"]), img_size, img_size, 3), np.uint8)
        hdf5_output.create_dataset("train_mean", (img_size, img_size, 3), np.float32)
        hdf5_output.create_dataset("train_std", (img_size, img_size, 3), np.float32)

    start_time = time()
    small_start = start_time
    for img_type, img_list in files_dict.items():
        for index, fileName in enumerate(img_list):
            im = Image.open(fileName)
            # Discard black and white images, breaks rest of pipeline
            if im.mode == 'RGB':
                # If its taller than it is wide, crop first
                if im.size[1] > im.size[0]:
                    crop_shift = random.randrange(im.size[1] - im.size[0])
                    im = im.crop(
                        (0, crop_shift, im.size[0], im.size[0] + crop_shift))
                elif im.size[0] > im.size[1]:
                    crop_shift = random.randrange(im.size[0] - im.size[1])
                    im = im.crop(
                        (crop_shift, 0, im.size[1] + crop_shift, im.size[1]))
                im = im.resize((img_size, img_size), resample=Image.LANCZOS)
                numpy_image = np.array(im, dtype=np.uint8)
                # Calculate per feature mean and variance on training data only
                if img_type == "train_img":
                    # Using Welford method for online calculation of mean and
                    # variance
                    if index > 0:
                        training_mean_new = training_mean_old + \
                                            (numpy_image - training_mean_old) / (index + 1)
                        training_variance = training_variance + \
                                            (numpy_image - training_mean_old) * \
                                            (numpy_image - training_mean_new)
                        training_mean_old = training_mean_new
                    else:
                        training_mean_new = numpy_image
                        training_mean_old = numpy_image
                # Save the image to the HDF5 output file
                hdf5_output[img_type][index, ...] = numpy_image
                if index % 1000 == 0 and index > 0:
                    small_end = time()
                    print("Saved {} {}s to hdf5 file in {} seconds".format(
                        index, img_type, small_end - small_start))
                    small_start = time()
    # Calculate the std using the variance array
    training_std = np.zeros((img_size, img_size, 3), dtype=np.float32)
    np.sqrt(training_variance / (len(files_dict["train_img"]) - 1), training_std)
    hdf5_output["train_mean"][...] = training_mean_new
    hdf5_output["train_std"][...] = training_std
    end_time = time()
    print("Elapsed time: {} seconds".format(end_time - start_time))


if __name__ == "__main__":
    directory = "/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/unlabeled2017/"
    # path to images; download link: http://images.cocodataset.org/zips/unlabeled2017.zip
    file_names = glob.glob(directory + "*.jpg")
    generate_data(file_names,
                  img_size=256,
                  TEST_SUBSET_DATA=False,
                  hdf5_file_name='COCO_2017_unlabeled.h5')
