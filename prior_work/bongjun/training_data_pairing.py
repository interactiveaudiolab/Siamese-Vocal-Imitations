import numpy as np
import librosa
import os
import pickle
import keras

from collections import Counter
import random

def normalize_spectrogram(data):
    '''
    data: (num_examples, num_freq_bins, num_time_frames)
    '''
    m = np.mean(data, axis=(1,2))
    m = m.reshape(m.shape[0], 1, 1)
    std = np.std(data, axis=(1,2))
    std = std.reshape(std.shape[0], 1, 1)
    
    m_matrix = np.repeat(np.repeat(m, data.shape[1], axis=1), data.shape[2], axis=2)
    std_matrix = np.repeat(np.repeat(std, data.shape[1], axis=1), data.shape[2], axis=2)
    
    data_norm = np.multiply(data - m_matrix, 1./std_matrix)
    
    return data_norm

def create_pairs(train_data, train_labels, ref_data, num_classes):
    '''
    create positive/negative pairs. 
    examples for negative pairs are randomly selected. 
    '''
    num_data_per_class = Counter(train_labels)
    n = min(num_data_per_class.values())
    pairs_left = []
    pairs_right = []
    pairs_labels = []
    #for data, label in zip(train_data, train_labels):
    for data, label in zip(train_data, train_labels):
        pairs_left.append(data)
        pairs_right.append(ref_data[label])
        pairs_labels.append(1)

        allowed_labels = list(range(0, num_classes))
        allowed_labels.remove(label)
        random_label = random.choice(allowed_labels)
        pairs_left.append(data)
        pairs_right.append(ref_data[random_label])
        pairs_labels.append(0)
    
    return np.array(pairs_left), np.array(pairs_right), np.array(pairs_labels) 

train_data = np.load('./data/npy/TL/first_half_train_X.npy')
train_labels = np.load('./data/npy/TL/first_half_train_Y.npy')

val_data = np.load('./data/npy/TL/first_half_val_X.npy')
val_labels = np.load('./data/npy/TL/first_half_val_Y.npy')

ref_data = np.load('./data/npy/TL/first_half_ref_X.npy')
ref_labels = np.load('./data/npy/TL/first_half_ref_Y.npy')

train_data_norm = normalize_spectrogram(train_data)
val_data_norm = normalize_spectrogram(val_data)
ref_data_norm = normalize_spectrogram(ref_data)

num_classes = ref_labels.shape[0]
train_pairs_left, train_pairs_right, train_pairs_labels = create_pairs(train_data_norm, train_labels, ref_data_norm, num_classes)
val_pairs_left, val_pairs_right, val_pairs_labels = create_pairs(val_data_norm, val_labels, ref_data_norm, num_classes)

print train_pairs_left.shape
print train_pairs_right.shape
print train_pairs_labels.shape

## Shuffle
idx = np.random.permutation(train_pairs_left.shape[0])
train_pairs_left_shuf, train_pairs_right_shuf, train_pairs_labels_shuf = train_pairs_left[idx], train_pairs_right[idx], train_pairs_labels[idx]

idx = np.random.permutation(val_pairs_left.shape[0])
val_pairs_left_shuf, val_pairs_right_shuf, val_pairs_labels_shuf = val_pairs_left[idx], val_pairs_right[idx], val_pairs_labels[idx]

# reshaping for Siamese network
tr_left_final = train_pairs_left_shuf.reshape(train_pairs_left_shuf.shape[0], 1, 39, 482)
tr_right_final = train_pairs_right_shuf.reshape(train_pairs_right_shuf.shape[0], 1, 128, 128)
val_left_final = val_pairs_left.reshape(val_pairs_left.shape[0], 1, 39, 482)
val_right_final = val_pairs_right.reshape(val_pairs_right.shape[0], 1, 128, 128)

# np.save('./data/npy/TL/train_pairs_data_left.npy', tr_left_final)
# np.save('./data/npy/TL/train_pairs_data_right.npy', tr_right_final)
# np.save('./data/npy/TL/train_pairs_labels.npy', train_pairs_labels_shuf)
# np.save('./data/npy/TL/val_pairs_data_left.npy', val_left_final)
# np.save('./data/npy/TL/val_pairs_data_right.npy', val_right_final)
# np.save('./data/npy/TL/val_pairs_labels.npy', val_pairs_labels)