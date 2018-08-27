# Siamese Networks to Evaluate Vocal Imitations
A program that trains up a siamese neural network that evaluates the likelihood that a vocal imitation is an imitation of a reference recording. Offers options to train with pairwise and triplet loss functions, as well as a variety of utilities that allow the user to perform replicable and reviewable trials.

## Prerequisites
All the software you need to get started is a python 3 installation and pip. If you want to train on the GPU, you also need to install CUDA.

## Installation
Clone the master branch of repository and install the dependencies (you should probably do this in a virtual environment):
```
pip install -r requirements.txt
```
Finally, fill in the values in `example_config.yaml` and rename it to `config.yaml`.

## Training
To see all training options:
``` 
python train.py -h
```
An example call that trains both pairwise and triplet loss models on 10 categories, using a fresh data split and set of weights, for 300 epochs and 20 trials, using CUDA:
``` 
python train.py -c -t -p -e 300 -tr 20 -rs -rw
```

## Datasets
This program can interact with two datasets released by the Interactive Audio Lab.
* [Vocal Imitation](https://github.com/interactiveaudiolab/VocalImitationSet) contains ~10 sound each of 302 sound concepts from [Google's AudioSet ontology](https://research.google.com/audioset/ontology/index.html). One of these references, deemed the "canonical" reference, has (on average) 18 vocal imitations of that sound.
    * [Zenodo download link](https://zenodo.org/record/1340763)
    * Because this dataset includes negative fine-grain pairs (e.g. an imitation of a dog barking paired with a different recording of a dog barking than the one that was imitated), it enables us to train using triplet loss.
* [Vocal Sketch](https://github.com/interactiveaudiolab/VocalSketchDataSet) contains 240 references with (on average) 28 imitations of each reference sound. Two versions of this dataset are available and supported; version 1.0 has the same amount of references but fewer imitations per reference and is a subset of version 1.1.
    * [Zenodo download link, v1.1](https://zenodo.org/record/1251982)
    * [Zenodo download link, v1.0](https://zenodo.org/record/164926)
    * This dataset cannot be used for training with triplet loss.

## Citation
TODO: insert paper citation

## Contact
Contact Brian Margolis (BrianMargolis2019 [at] u.northwestern.edu) with any questions regarding this work. 