import tensorflow as tf
from JPS import JigsawPuzzleSolver
from FT import FileTransfer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--task', help='Mode with which to run main: jps or fn (Jiggle Puzzle Solver or Fine Tuning)', required=True)
args = vars(parser.parse_args())
task = str(args['task'])

if task == 'jps':
    from config import conf
elif task == 'ft':
    from config_ft import conf_ft
else:
    raise ValueError("Mode is not well defined. Please select one from jps and ft")

tf.enable_eager_execution()


def JPS(conf):
    model = JigsawPuzzleSolver(conf)
    model.train()
    model.evaluate()


def FT(conf_ft):
    model = FileTransfer(conf_ft)
    # model.train()


if __name__ == '__main__':
    if task == 'jps':
        JPS(conf)
    elif task == 'ft':
        FT(conf_ft)
