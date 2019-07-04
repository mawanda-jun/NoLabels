import h5py
import tensorflow as tf
from Dataset.generate_hamming_set import hamming_set
from config import conf
import numpy as np
import os
from utils.logger import set_logger

if conf.model == 'alexnet':
    from nets.AlexNet.Siamese import Siamese_AlexNet as SiameseNet
else:
    raise ValueError('Please specify "alexnet" since no other net type has been implemented yet')


def main(_):
    if conf.generateHammingSet:
        hamming_set(conf.numCrops, conf.hammingSetSize,
                    conf.selectionMethod, conf.hammingFileName)

    with h5py.File(os.path.join('Dataset', conf.resources, conf.hammingFileName + str(conf.hammingSetSize) + '.h5'), 'r') as h5f:
        HammingSet = np.array(h5f['max_hamming_set'])

    if conf.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', conf.mode)
        print("Please input a mode: train, test, or predict")
    else:
        model = SiameseNet(tf.Session(), conf, HammingSet)
        os.makedirs(conf.modeldir + conf.run_name, exist_ok=True)
        os.makedirs(conf.logdir + conf.run_name, exist_ok=True)
        os.makedirs(conf.savedir + conf.run_name, exist_ok=True)
        set_logger(conf.modeldir + 'train_validation.log')
        if conf.mode == 'train':
            model.train()
        elif conf.mode == 'test':
            model.test(epoch_num=6)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    tf.app.run()
