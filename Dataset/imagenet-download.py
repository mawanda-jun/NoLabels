import imagedownloader
from tqdm import tqdm

if __name__ == '__main__':
    downloader = imagedownloader.ImageNetDownloader()

    f = open("imagenet_synset.txt", "r")
    father = ""
    for line in tqdm(f):
        wnid = line.replace("\n", "")
        print("\n\n" + wnid)
        if wnid.find('-') != -1:
            wnid = wnid.replace("-", "")
            list = downloader.getImageURLsOfWnid(wnid)
            downloader.downloadImagesByURLs(wnid, list, father)
        else:
            father = wnid
            list = downloader.getImageURLsOfWnid(wnid)
            downloader.downloadImagesByURLs(wnid, list, "")
    f.close()