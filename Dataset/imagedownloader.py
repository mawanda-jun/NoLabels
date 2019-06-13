import os
import urllib.request
import urllib.parse as urlparse
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


class ThreadedDownload(threading.Thread):
    def __init__(self, name, url_list, dir_path):
        threading.Thread.__init__(self)
        self.name = name
        self.url_list = url_list
        self.dir_path = dir_path

    def run(self):
        failed = 0
        for url in self.url_list:
            try:
                ImageNetDownloader.download_file(url, self.dir_path)
            except Exception:
                # print('Fail to download : ' + url)
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
    def getImageURLsOfWnid(wnid):
        url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + str(wnid)
        f = urllib.request.urlopen(url)
        contents = f.read().decode().split('\n')
        imageUrls = []

        for each_line in contents:
            each_line = each_line.replace('\r', '').strip()
            if each_line:
                imageUrls.append(each_line)

        return imageUrls

    def downloadImagesByURLs(self, wnid, imageUrls, father):
        if father != "":
            # print("Sono diverso "+father)
            wnid_urlimages_dir = os.path.join(self.mkWnidDir(father), wnid)
        else:
            # print("No difference")
            wnid_urlimages_dir = os.path.join(self.mkWnidDir(wnid))
        # print(wnid_urlimages_dir)
        # if not os.path.exists(wnid_urlimages_dir):
        #     os.makedirs(wnid_urlimages_dir)

        n_threads = 32

        chunked_list = chunks(imageUrls, n_threads)
        thread_list = []
        for i, c_list in enumerate(chunked_list):
            thread_list.append(ThreadedDownload(str(i), c_list, wnid_urlimages_dir))

        for thread in thread_list:
            thread.start()

    @staticmethod
    def mkWnidDir(wnid):
        wnid = os.path.join('images', wnid)
        if not os.path.exists(wnid):
            os.makedirs(wnid)
        return os.path.abspath(wnid)