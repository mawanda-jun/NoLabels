import os
import urllib.request
import urllib.parse as urlparse
import threading
from itertools import islice
from tqdm import tqdm
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
        failed = 0
        for line in self.url_list:
            url = line.split('\t')[1]
            url = url.replace('\n', '')
            url = url.strip()
            try:
                ImageNetDownloader.download_file(url, self.dir_path)

            except Exception:
                print('Fail to download : ' + url)
                failed += 1
        print("Thread: {name} \nDownloaded images: {images} \nFailed: {failed}".format(
            name=self.name,
            images=len(self.url_list) - failed,
            failed=failed)
        )


class ImageNetDownloader:
    def __init__(self):
        self.host = 'http://www.image-net.org'

    @staticmethod
    def download_file(url, desc=None, renamed_file=None):
        u = urllib.request.urlopen(url)

        scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
        filename = os.path.basename(path)
        if not filename:
            filename = 'downloaded.file'

        if not renamed_file is None:
            filename = renamed_file

        if desc:
            filename = os.path.join(desc, filename)

        with open(filename, 'wb') as f:
            meta = u.info()
            meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
            meta_length = meta_func("Content-Length")
            file_size = None
            if meta_length:
                file_size = int(meta_length[0])
            # print("Downloading: {0} Bytes: {1}".format(url, file_size))

            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)

                status = "{0:16}".format(file_size_dl)
                if file_size:
                    status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
                status += chr(13)

        return filename

    @staticmethod
    def downloadImagesByLines(lines):

        dir = os.path.join(os.getcwd(), 'images')
        n_threads = 32

        chunked_list = chunks(lines, n_threads)
        thread_list = []
        for i, c_list in enumerate(chunked_list):
            thread_list.append(ThreadedDownload(str(i), c_list, dir))

        for thread in thread_list:
            thread.start()


if __name__ == '__main__':
    dir = os.path.join(os.getcwd(), 'images')
    os.makedirs(dir, exist_ok=True)  # if present doesn't do anything
    # count = 0
    totImages = 500          # 500K

    with open('fall11_urls.txt', 'r', encoding='utf-8') as f:
        # while count < totImages:
        #     dimLines = 100
        #     countLines = 0
        #     lines = []
        #     for line in f:
        #         lines.append(line)
        #         countLines = countLines + 1
        #         if countLines == dimLines:
        #             break
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[0:totImages]
    ImageNetDownloader.downloadImagesByLines(lines)
