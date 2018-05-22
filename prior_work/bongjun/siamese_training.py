import numpy as np
import librosa
import os
import pickle
import keras
# import matplotlib.pyplot as plt
from collections import Counter
import random

np.random.seed(1337)

from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.optimizers import RMSprop, SGD, Adadelta, Adagrad
from keras import regularizers       
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils, generic_utils
from keras import backend as K
K.set_image_dim_ordering('th')

#os.environ["CUDA_VISIBLE_DEVICES"]="1"

train_data_left = np.load('./npy/train_pairs_data_left.npy')
train_data_right = np.load('./npy/train_pairs_data_right.npy')
train_labels = np.load('./npy/train_pairs_labels.npy')
val_data_left = np.load('./npy/val_pairs_data_left.npy')
val_data_right = np.load('./npy/val_pairs_data_right.npy')
val_labels = np.load('./npy/val_pairs_labels.npy')

print train_data_left.shape

# Left branch: vocal imitaiton
input_a = Input(shape=(1, 39, 482),name='input_a')

x = Conv2D(48, (6, 6), padding='valid',  name='lid_conv_1')(input_a)
x = BatchNormalization(axis=1,name='lid_conv_1_BN')(x)
x = Activation('relu', name='lid_conv_1_AC')(x)
x = MaxPooling2D(pool_size=(2, 2), name='lid_conv_1_MP')(x)

x = Conv2D(48, (6, 6), padding='valid',  name='lid_conv_2')(x)
x = BatchNormalization(axis=1, name='lid_conv_2_BN')(x)
x = Activation('relu',name='lid_conv_2_AC')(x)
x = MaxPooling2D(pool_size=(2, 2), name='lid_conv_2_MP')(x)

x = Conv2D(48, (6, 6), padding='valid',  name='lid_conv_3')(x)
x = BatchNormalization(axis=1, name='lid_conv_3_BN')(x)
x = Activation('relu', name='lid_conv_3_AC')(x)
x = MaxPooling2D(pool_size=(1, 2), name='lid_conv_3_MP')(x)


x = Flatten(name='lid_conv_3_flatten')(x)

output_a = x
model_a = Model(input_a, output_a)

# for l in model_a.layers:
#     l.trainable = False

# Right branch: reference recording
input_b = Input(shape=(1, 128, 128), name='input_b')

y = Conv2D(24, (5, 5), padding='valid',  name='nyu_conv_1')(input_b)
y = BatchNormalization(axis=1, name='nyu_conv_1_BN')(y)
y = Activation('relu', name='nyu_conv_1_AC')(y)
y = MaxPooling2D(pool_size=(2, 4), name='nyu_conv_1_MP')(y)

y = Conv2D(48, (5, 5), padding='valid', name='nyu_conv_2')(y)
y = BatchNormalization(axis=1, name='nyu_conv_2_BN')(y)
y = Activation('relu', name='nyu_conv_2_AC')(y)
y = MaxPooling2D(pool_size=(2, 4), name='nyu_conv_2_MP')(y)

y = Conv2D(48, (5, 5), padding='valid', name='nyu_conv_3')(y)
y = BatchNormalization(axis=1, name='nyu_conv_3_BN')(y)
y = Activation('relu', name='nyu_conv_3_AC')(y)

y = Flatten(name='nyu_conv_3_flatten')(y)

output_b = y
model_b = Model(input_b, output_b)

# for l in model_b.layers:
#     l.trainable = False

print train_data_right.shape
print train_labels.shape

#print 'model loadning'
#model_a.load_weights('./data/speech/lid_cnn_weights_69_8_bkedit.h5', by_name=True)
#model_b.load_weights('./data/urbansound8k/bk_training/weights_fold05.12-0.78_test.h5',by_name=True)

# Then instantiate the imitation and recording model
imi_input_a = Input(shape=(1, 39, 482))
ref_input_b = Input(shape=(1, 128, 128))

imi_output_a = model_a(imi_input_a)
ref_output_b = model_b(ref_input_b)

concatenated = keras.layers.concatenate([imi_output_a, ref_output_b])

out = Dense(108, activation = 'relu')(concatenated)
out = Dense(1, activation='sigmoid')(out)

overall_model = Model(inputs = [imi_input_a, ref_input_b], outputs = out)

#overall_model.load_weights('./TL-IMINET_ICASSP2018_Code/Stage3_Finetuning/model/with_pretraining/with_pretrain_asym_metric.h5')
#overall_model.load_weights('./TL-IMINET_ICASSP2018_Code/Stage3_Finetuning/model/without_pretraining/without_pretrain_asym_metric.h5')
#overall_model.load_weights('./with_pretrain_asym_metric.h5')

sgd = keras.optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)
overall_model.compile(loss='binary_crossentropy',
                      optimizer=sgd,
                      metrics=['acc'])

history = overall_model.fit([train_data_left, train_data_right], train_labels,
          validation_data=([val_data_left, val_data_right], val_labels),
          batch_size=128,
          epochs=70, 
          #callbacks=callback, 
          shuffle=True,
          #sample_weight = weights
          )