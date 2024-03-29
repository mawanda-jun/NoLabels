import os
from PIL import Image
from tqdm import tqdm
import threading


def chunks(l: list, n: int):
    """
    Divide a list into n equal parts
    :param l: list
    :param n: number of chunks
    :return: list of list of the original
    """
    k, m = divmod(len(l), n)
    return (l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class ThreadIt(threading.Thread):
    def __init__(self, filepaths, mode='too_little'):
        threading.Thread.__init__(self)
        self.filepaths = filepaths
        self.mode = mode

    def run(self):
        if self.mode == 'too_little':
            for line in tqdm(self.filepaths, desc="Deleting error images"):
                # newline = line.split(' ')[4]
                # newline = newline.replace("'", "")
                # newline = newline.replace('\\\\', "\\")
                line = line.replace('\n', '').replace('\'', '/')
                path = os.path.join(line)
                if os.path.isfile(path):
                    os.remove(path)


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
    too_little = set([])
    for root, dirs, files in os.walk(path_to_folder):
        for file in tqdm(files, desc='Finding unopenable images'):
            try:
                # print(file)
                img = Image.open(os.path.join(root, file))
                if img.height < 256 or img.width < 256 or img.mode is not 'RGB':
                    too_little.add(os.path.join(root, file + '\n'))
            except Exception as e:
                errors.add(str(e) + '\n')
    with open('errors.txt', 'w') as f:
        f.writelines(errors)
    with open('too_little.txt', 'w') as f:
        f.writelines(too_little)


def delete_images(errors_filename, too_little_filename):
    """
    Delete all images that the algorithm in "create_h5py_dataset" has not been able to open
    :param errors_filename:
    :return:
    """
    with open(errors_filename, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Deleting error images"):
            newline = line.split(' ')[4]
            newline = newline.replace("'", "")
            newline = newline.replace('\\\\', "\\")
            newline = newline.replace('\n', '')
            path = os.path.join(newline)
            if os.path.isfile(path):
                os.remove(path)
    with open(too_little_filename, 'r') as f:
        for line in tqdm(f.readlines(), desc="Deleting too little images"):
            # if os.path.isfile(line.replace('resources/', '')):
            os.remove(os.path.join(line.replace('resources\'', '').replace('\n', '').replace('\\', '/')))


if __name__ == '__main__':
    # dirpath = os.path.join('resources', 'images')
    dirpath = os.path.join('C:\\Users\\giova\\Desktop\\ILSVRC\\Data\\CLS-LOC')
    # rename_images(dirpath)
    find_unopenable(dirpath)
    # delete_images('errors.txt', 'too_little.txt')
