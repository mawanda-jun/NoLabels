import os
from PIL import Image


def rename_images(folder_path):
    """
    Rename all images in folder that does not contain the enabled extensions
    :param folder_path:
    :return:
    """
    folder_path = os.path.join(os.getcwd(), folder_path)
    enabled_extensions = ['.bmp', '.png', '.jpg', '.JPG', '.jpeg', '.PNG', '.JPEG', '.BMP']

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            flag = False
            for extension in enabled_extensions:
                if extension in os.path.splitext(file)[1]:
                    flag = True
            if not flag:
                os.rename(os.path.join(os.getcwd(), root, file), os.path.join(os.getcwd(), root, file + '.jpeg'))


def find_unopenable(path_to_folder):
    errors = set([])
    for root, dirs, files in os.walk(path_to_folder):
        for file in files:
            try:
                Image.open(os.path.join(root, file))
            except OSError as e:
                errors.add(str(e) + '\n')
    with open('errors.txt', 'w') as f:
        f.writelines(errors)


def delete_images(errors_filename):
    """
    Delete all images that the algorithm in "create_h5py_dataset" has not been able to open
    :param errors_filename:
    :return:
    """
    with open(errors_filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            newline = line.split(' ')[4]
            newline = newline.replace("'", "")
            newline = newline.replace('\\\\', "\\")
            newline = newline.replace('\n', '')
            path = os.path.join(newline)
            if os.path.isfile(path):
                os.remove(path)


if __name__ == '__main__':
    dirpath = os.path.join('resources', 'images')
    rename_images(dirpath)
    find_unopenable(dirpath)
    delete_images('errors.txt')
