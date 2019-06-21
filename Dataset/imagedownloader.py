import os
import urllib.request
import urllib.parse as urlparse
import threading
import random


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
            if i % 100 == 0:
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
        # u = urllib.request.urlopen(url)

        # scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        # filename = os.path.basename(path)
        filename = os.path.basename(url)
        if not filename:
            filename = 'downloaded.jpeg'

        if renamed_file is not None:
            filename = renamed_file
        if not os.path.isfile(os.path.join(desc, filename)):
            try:
                urllib.request.urlretrieve(url, os.path.join(desc, filename))
            except Exception:
                with open(os.path.join(desc, filename), 'w') as f:
                    f.write('File not found')
        # if desc:
        #     filename = os.path.join(desc, filename)
        # if not os.path.isfile(filename):
        #     with open(filename, 'wb') as f:
        #         meta = u.info()
        #         meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        #         meta_length = meta_func("Content-Length")
        #         file_size = None
        #         if meta_length:
        #             file_size = int(meta_length[0])
        #         # print("Downloading: {0} Bytes: {1}".format(url, file_size))
        #
        #         file_size_dl = 0
        #         block_sz = 8192
        #         while True:
        #             buffer = u.read(block_sz)
        #             if not buffer:
        #                 break
        #
        #             file_size_dl += len(buffer)
        #             f.write(buffer)
        #
        #             status = "{0:16}".format(file_size_dl)
        #             if file_size:
        #                 status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
        #             status += chr(13)

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
    n_threads = 30
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
