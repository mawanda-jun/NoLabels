# NoLabels
This project is a TensorFlow eager implementation of the paper [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246).
The target task is based on the classification competition of [Food 101](https://www.kaggle.com/kmader/food41).

## Getting started with the Jigsaw puzzle training
First of all, create a virtual environment and install all the requirements listed in `requirements.txt`. We suggest to use TensorFlow GPU, while the version v | 1.14<= v < 2.0 is mandatory.
Here there is a list of the files that have to be configured in order to run the Jigsaw puzzle solver task:
1. `relazione/paper.pdf`: contains the theory and practical description of this project (ENG);
2. `Dataset/create_dataset_h5.py`: is the file that creates the dataset that the Jigsaw puzzle solver with use for the pretext task;
The project can be implemented with every kind of images, but this project has been tested with the [ILSVRC2017](http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php) CLS-LOC dataset and `500K` images. The number of desired images and the resource folder path can be configured inside this `.py` in the very first lines. This python file will create a `n_el` number of images under `Dataset/resources/images` divided in `train`, `val` and `test` sets and a `.h5` file in the same directory containing the mean, std and dimension of the dataset;
3. `config.py`: is the configuration file for the Jigsaw Puzzle pretext task. In order to start the execution some parameters shall be changed:
    - `hammingSetSize`: defines the number of permutations to be used. We set it to 40 as the task should be too difficult;
    - `data_path`: this is the folder of the dataset. The root of this folder must contain the `train`, `val` and `test` images together with their `.h5` description file. If you have generated the dataset with `create_dataset_h5.py` unchanged then you don't have to change this parameter;
    - the other parameters are all tunable, altough this implementation has not been tested with different Jigsaw parameters.
4. `Dataset/generate_hamming_set.py`: this file generates the hamming set that is needed for the training. Just run it to create the `max_hamming_x.h5` file;
5. run the `main.py` file. It accepts two `--mode` parameters: `jps` and `ft`, that select which training the user want to do.
