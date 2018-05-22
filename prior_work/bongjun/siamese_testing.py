import pickle
#import matplotlib.pyplot as plt
#%matplotlib inline
import os

import numpy as np
import librosa
from keras.models import Model
from keras import layers
from keras import Input
from keras import optimizers
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

np.random.seed(1337)

def compute_MRR(test_ref_labels, test_imi_labels, model_output):
    num_classes = test_ref_labels.shape[0]
    num_queries = test_imi_labels.shape[0]

    prediction_matrix = model_output.reshape((num_classes, num_queries), order='F')

    overall_ranking = []
    for i, cur_label in enumerate(test_imi_labels):

        pred_per_query = prediction_matrix[:, i]

        idx = np.argsort(pred_per_query)[::-1]
        pred_per_query_sorted = pred_per_query[idx]
        labels_sorted = test_ref_labels[idx]

        cur_rank = np.where(labels_sorted == cur_label)[0][0]
        overall_ranking.append(cur_rank + 1)
    overall_ranking = np.array(overall_ranking)
    MRR = np.sum(1./overall_ranking)/len(overall_ranking)


    return MRR


def compute_recall(test_ref_labels, test_imi_labels, model_output, k):
    num_classes = test_ref_labels.shape[0]
    num_queries = test_imi_labels.shape[0]

    prediction_matrix = model_output.reshape((num_classes, num_queries), order='F')

   
    #overall_ranking = []
    overall_recall = []
    for i, cur_label in enumerate(test_imi_labels):

        pred_per_query = prediction_matrix[:, i]

        idx = np.argsort(pred_per_query)[::-1]
        pred_per_query_sorted = pred_per_query[idx]
        labels_sorted = test_ref_labels[idx]

        cur_rank = np.where(labels_sorted == cur_label)[0][0]
        if cur_rank + 1 <=k:
            overall_recall.append(1)
        else:
            overall_recall.append(0)
        #overall_ranking.append(cur_rank + 1)
    overall_recall = np.array(overall_recall)
    mean_recall = np.mean(overall_recall)
   
    return mean_recall


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

def create_pairs_for_retreival_test(test_imi_data, test_imi_labels, ref_data, ref_labels):

    pairs_left = []
    pairs_right = []
    pairs_labels = []
    for imi, imi_label in zip(test_imi_data, test_imi_labels):
        for ref, ref_label in zip(ref_data, ref_labels):
            pairs_left.append(imi)
            pairs_right.append(ref)
            if imi_label == ref_label:
                pairs_labels.append(1)
            else:
                pairs_labels.append(0)
    return np.array(pairs_left), np.array(pairs_right), np.array(pairs_labels) 



# # Measure MRR on Testing set (cross-category MRR)
test_imi_data = np.load('./npy/second_half_test_X_ALL.npy')
test_imi_labels = np.load('./npy/second_half_test_Y_ALL.npy')
test_ref_data = np.load('./npy/second_half_ref_X_ALL.npy')
test_ref_labels = np.load('./npy/second_half_ref_Y_ALL.npy')

test_imi_data_norm = normalize_spectrogram(test_imi_data)
test_ref_data_norm = normalize_spectrogram(test_ref_data)


test_retr_pairs_left, test_retr_pairs_right, test_retr_pairs_labels = create_pairs_for_retreival_test(test_imi_data_norm, test_imi_labels, test_ref_data_norm, test_ref_labels)

test_retr_pairs_left = test_retr_pairs_left.reshape(test_retr_pairs_left.shape[0], 1, 39, 482)
test_retr_pairs_right = test_retr_pairs_right.reshape(test_retr_pairs_right.shape[0], 1, 128, 128)

model_path = './model/model_11-10_top_pair.h5'

overall_model = load_model(model_path)

model_output = overall_model.predict([test_retr_pairs_left, test_retr_pairs_right],  batch_size=128, verbose=1)

MRR = compute_MRR(test_ref_labels, test_imi_labels, model_output)

print 'MRR(ALL): ', MRR


# # Measure MRR on Testing set (within-category MRR)
for test_category in ['ai', 'cs', 'ss', 'ed']:
    print test_category

    test_imi_data = np.load('./npy/second_half_test_X_'+test_category+'.npy')
    test_imi_labels = np.load('./npy/second_half_test_Y_'+test_category+'.npy')
    test_ref_data = np.load('./npy/second_half_ref_X_'+test_category+'.npy')
    test_ref_labels = np.load('./npy/second_half_ref_Y_'+test_category+'.npy')

    test_imi_data_norm = normalize_spectrogram(test_imi_data)
    test_ref_data_norm = normalize_spectrogram(test_ref_data)

    test_retr_pairs_left, test_retr_pairs_right, test_retr_pairs_labels = create_pairs_for_retreival_test(test_imi_data_norm, test_imi_labels, test_ref_data_norm, test_ref_labels)

    test_retr_pairs_left = test_retr_pairs_left.reshape(test_retr_pairs_left.shape[0], 1, 39, 482)
    test_retr_pairs_right = test_retr_pairs_right.reshape(test_retr_pairs_right.shape[0], 1, 128, 128)

    #model_path = './model/model_11-10_top_pair.h5'
    overall_model = load_model(model_path)

    model_output = overall_model.predict([test_retr_pairs_left, test_retr_pairs_right],  batch_size=128, verbose=1)

    MRR = compute_MRR(test_ref_labels, test_imi_labels, model_output)

    print test_category
    print 'MRR: ', MRR
