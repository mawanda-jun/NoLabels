import tensorflow as tf
from JPS import JigsawPuzzleSolver
from FT import FileTransfer
import argparse
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='Mode with which to run main: jps or fn (Jiggle Puzzle Solver or Fine Tuning)', required=True)
args = vars(parser.parse_args())
mode = str(args['mode'])

if mode == 'jps':
    from config import conf
elif mode == 'ft':
    from config_ft import conf_ft


def JPS(conf):
    model = JigsawPuzzleSolver(conf)
    model.train()


def FT(conf_ft):
    model = FileTransfer(conf_ft)
    # model.train()


if __name__ == '__main__':
    if mode == 'jps':
        JPS(conf)
    elif mode == 'fn':
        FT(conf_ft)
