import h5py
import tensorflow as tf
from Dataset.generate_hamming_set import hamming_set
from config import args
import numpy as np
import os

if args.model == 'alexnet':
    from nets.AlexNet.Siamese import Siamese_AlexNet as SiameseNet
else:
    from nets.CapsNet.Siamese import SiameseCapsNet as SiameseNet


def main(_):
    if args.generateHammingSet:
        hamming_set(args.numCrops, args.hammingSetSize,
                    args.selectionMethod, args.hammingFileName)

    with h5py.File(os.path.join('Dataset', args.resources, args.hammingFileName+str(args.hammingSetSize)+'.h5'), 'r') as h5f:
        HammingSet = np.array(h5f['max_hamming_set'])

    if args.mode not in ['train', 'test', 'predict']:
        print('invalid mode: ', args.mode)
        print("Please input a mode: train, test, or predict")
    else:
        model = SiameseNet(tf.Session(), args, HammingSet)
        os.makedirs(args.modeldir+args.run_name, exist_ok=True)
        os.makedirs(args.logdir+args.run_name, exist_ok=True)
        os.makedirs(args.savedir+args.run_name, exist_ok=True)
        if args.mode == 'train':
            model.train()
        elif args.mode == 'test':
            model.test(epoch_num=6)


if __name__ == '__main__':
    # configure which gpu or cpu to use
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    tf.app.run()
