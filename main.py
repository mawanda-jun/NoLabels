import tensorflow as tf
from JPS import JigsawPuzzleSolver
from config import conf
import argparse
tf.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='Mode with which to run main: jps or fn (Jiggle Puzzle Solver or Fine Tuning)', required=True)
args = vars(parser.parse_args())
mode = str(args['mode'])


def JPS(conf):
    model = JigsawPuzzleSolver(conf)
    model.train()


if __name__ == '__main__':
    if mode == 'jps':
        JPS(conf)
    elif mode == 'fn':
        pass
