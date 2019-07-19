# NoLabels
This project is a TensorFlow eager implementation of the paper [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246).
The target task is based on the classification competition of [Food 101](https://www.kaggle.com/kmader/food41).

## Getting started
The repository is composed of various folders and files. Here there is a list of the files that have to be configured in order to run the project:
- `relazione/paper.pdf`: contains the theory and practical description of this project (ENG);
- `Dataset/create_dataset_h5.py`: is the file that creates the dataset that the Jigsaw puzzle solver with use for the pretext task.
The project can be implemented with every kind of images, but this project has been tested with the [ILSVRC2017](http://image-net.org/challenges/LSVRC/2017/download-images-1p39.php) CLS-LOC dataset and
`500K` images. The number of desired images can be configured inside this `.py` file. This python file will create a `n_el` 
number of images under `Dataset/resources/images` and a `.h5` file in the same directory containing the mean, std and dimension of the dataset.
- `config.py`: is the configuration file for the Jigsaw Puzzle pretext task. There are many 
