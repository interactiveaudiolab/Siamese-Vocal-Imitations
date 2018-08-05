# Siamese Networks to Evaluate Vocal Imitations
A program that trains up a siamese neural network that evaluates the likelihood that a vocal imitation is an imitation of a reference recording. Offers options to train with pairwise and triplet loss functions, as well as a variety of utilities that allow the user to perform replicable and reviewable trials.

## Use
Clone the repository, and then in the parent directory:
```
pip install -r requirements.txt
```
To see all training options:
``` 
python experiment.py -h
```
An example call that trains both pairwise and triplet loss models on 10 categories, using a fresh data split and set of weights, for 300 epochs and 20 trials, using CUDA:
``` 
python experiment.py -c -t -p -e 300 -tr 20 -rs -rw
```

## Citation
TODO: insert paper citation


## Contact
Contact Brian Margolis (BrianMargolis2019 [at] u.northwestern.edu) with any questions or issues regarding this work.