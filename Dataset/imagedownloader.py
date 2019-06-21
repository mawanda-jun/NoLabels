import os
import urllib
from urllib.error import URLError
import urllib.request
import threading
import random
from PIL import Image


def chunks(l: list, n: int):
    """
    Divide a list into n equal parts
    :param l: list
    :param n: number of chunks
    :return: list of list of the original
    """
    k, m = divmod(len(l), n)
    return (l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


class ThreadedDownload(threading.Thread):
    def __init__(self, name, lines_list, dir_path):
        threading.Thread.__init__(self)
        self.name = name
        self.url_list = lines_list
        self.dir_path = dir_path

    def run(self):
        for i, line in enumerate(self.url_list):
            if i % 10 == 0:
                print('Thread {n} processed {i} urls'.format(n=self.name, i=i))
            url = line.split('\t')[1]
            url = url.replace('\n', '')
            url = url.strip()
            ImageNetDownloader.download_file(url, self.dir_path, self.name + '_' + str(i) + '.jpeg')
        print("Thread: {name} \nDownloaded images: {images}".format(
            name=self.name,
            images=len(self.url_list)))


class ImageNetDownloader:
    def __init__(self):
        self.host = 'http://www.image-net.org'

    @staticmethod
    def download_file(url, desc=None, renamed_file=None):
        filename = os.path.basename(url)
        if not filename:
            filename = 'downloaded.jpeg'

        if renamed_file is not None:
            filename = renamed_file
        if not os.path.isfile(os.path.join(desc, filename)):
            try:
                urllib.request.urlretrieve(url, os.path.join(desc, filename))
            except URLError:
                with open(os.path.join(desc, filename), 'w') as f:
                    f.write('File not found')
        else:
            # when resuming download, this check if images are available this time
            try:
                Image.open(os.path.join(desc, filename))
            except OSError:
                try:
                    urllib.request.urlretrieve(url, os.path.join(desc, filename))
                    print('Recovered image {}'.format(filename))
                except URLError:
                    with open(os.path.join(desc, filename), 'w') as f:
                        f.write('File not found')

        return filename

    @staticmethod
    def downloadImagesByLines(lines, dest_path, n_threads):
        dirname = os.path.join(os.getcwd(), dest_path, 'images')
        os.makedirs(dirname, exist_ok=True)  # if present doesn't do anything

        chunked_list = chunks(lines, n_threads)
        thread_list = []
        for i, c_list in enumerate(chunked_list):
            thread_list.append(ThreadedDownload(str(i), c_list, dirname))

        for thread in thread_list:
            thread.start()


if __name__ == '__main__':
    # count = 0
    new_urls = 'new_links.txt'
    dest_path = 'resources'
    n_threads = 2
    tot_images = int(2e5)
    random.seed(3)  # force shuffle order
    if not os.path.isfile(new_urls):
        with open(os.path.join(dest_path, 'fall11_urls.txt'), 'r', encoding='latin-1') as f:
            lines = f.readlines()
            random.shuffle(lines)
            lines = lines[0:tot_images]
            with open(os.path.join(dest_path, new_urls), 'w', encoding='latin-1') as w:
                w.writelines(lines)

    with open(os.path.join(dest_path, new_urls), 'r', encoding='latin-1') as f:
        ImageNetDownloader.downloadImagesByLines(f.readlines(), dest_path, n_threads)
