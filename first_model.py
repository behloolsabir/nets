import os
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Dropout, merge, Flatten, Reshape, MaxPooling2D, Convolution2D, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

# def single_val(input_layer=None, out_dim=0):
    # dense1 = Dense(2048)(input_layer)
    # dense2 = Dense(512)(dense1)
    # dense3 = Dense(out_dim, W_regularizer='l2', activation='tanh')(dense2)
    # return dense3

def multi_dim(feat_dim_array=300, input_layer=None, out_dim=0):
    cnn_layer1 = Convolution2D(64,3,feat_dim_array, activation='relu', border_mode='valid')(input_layer)
    cnn_layer2 = Convolution2D(64,3,1, activation='relu', border_mode='same')(cnn_layer1)
    cnn_layer3 = Convolution2D(64,3,1, activation='relu', border_mode='same')(cnn_layer2)
    max_pool_layer = MaxPooling2D((2,1))(cnn_layer3)
    flat_layer = Flatten()(max_pool_layer)
    dense1 = Dense(out_dim, W_regularizer='l2')(flat_layer)
    return dense1

def single_dim(input_layer=None, out_dim=0):
    # print '////////////////////', input_layer.get_shape()[-1].value
    in_dim = input_layer.get_shape()[-1].value
    dense1 = Dense(in_dim*20, W_regularizer='l2', activation='relu')(input_layer)
    dense2 = Dense(in_dim*10, W_regularizer='l2', activation='relu')(dense1)
    dense3 = Dense(out_dim, W_regularizer='l2', activation='relu')(dense2)
    return dense3



def fc_cnn_model(input_dim_array, feat_dim_array):
    fc_recursive_output = []
    single_dim_output = []
    input_layers = []
    out_dim = 512
    single_out_dim = 128
    for i in range(len(input_dim_array)):
        if feat_dim_array[i] == 0:
            input_layer = Input(shape=(input_dim_array[i],))
            input_layers.append(input_layer)
            # out_layer = single_val(input_layer, out_dim)
            fc_recursive_output.append(out_layer)
        elif feat_dim_array[i] == 1:
            input_layer = Input(shape=(input_dim_array[i],))
            input_layers.append(input_layer)
            out_layer = single_dim(input_layer, single_out_dim)
            out_layer = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(out_layer)
            single_dim_output.append(out_layer)
        elif feat_dim_array[i] == 300 and input_dim_array[i] == 1:
            input_layer = Input(shape=(input_dim_array[i],feat_dim_array[i]))
            input_layers.append(input_layer)
            input_layer = Reshape((feat_dim_array[i],))(input_layer)
            out_layer = single_dim(input_layer, single_out_dim)
            out_layer = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(out_layer)
            single_dim_output.append(out_layer)
        # else :
        elif input_dim_array[i] < 10000:
            input_layer = Input(shape=(input_dim_array[i], feat_dim_array[i]))
            input_layers.append(input_layer)
            reshape_layer = Reshape((input_dim_array[i],feat_dim_array[i],1))(input_layer)
            out_layer = multi_dim(feat_dim_array[i], reshape_layer, out_dim)
            out_layer = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(out_layer)
            fc_recursive_output.append(out_layer)


    


    single_merged_vector = merge(single_dim_output, mode='concat', concat_axis=-1)
    reshape_layer = Reshape((len(single_dim_output), single_out_dim, 1))(single_merged_vector)
    cnn_layer3 = Convolution2D(32,3,3, activation='relu', border_mode='valid')(reshape_layer)
    max_pool_layer2 = MaxPooling2D((2,2))(cnn_layer3)
    flat_layer1 = Flatten()(max_pool_layer2)
    
    merged_vector = merge(fc_recursive_output, mode='concat', concat_axis=-1)
    reshape_layer = Reshape((out_dim, len(fc_recursive_output), 1))(merged_vector)
    cnn_layer3 = Convolution2D(32,3,3, activation='relu', border_mode='valid')(reshape_layer)
    max_pool_layer2 = MaxPooling2D((2,2))(cnn_layer3)
    flat_layer2 = Flatten()(max_pool_layer2)

    merged_vector = merge([flat_layer1, flat_layer2], mode='concat', concat_axis=-1)

    dense_layer_1 = Dense(128, W_regularizer='l2', activation='relu')(merged_vector)
    dense_layer_2 = Dense(128, W_regularizer='l2', activation='relu')(dense_layer_1)
    dense_layer_3 = Dense(64, W_regularizer='l2', activation='relu')(dense_layer_2)
    dense_layer_4 = Dense(64, W_regularizer='l2', activation='relu')(dense_layer_3)
    dense_layer_5 = Dense(16, W_regularizer='l2', activation='relu')(dense_layer_4)
    dense_layer_6 = Dense(16, W_regularizer='l2', activation='relu')(dense_layer_5)
    out_layer = Dense(2, activation='softmax')(dense_layer_6)
    # out_layer = Dense(1, activation='softmax')(dense_layer_6)
    model = Model(input=input_layers, output=out_layer)
    # optimizer = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
    # optimizer = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
    # optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    # optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    optimizer = Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    # optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.001)



    model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    # model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    return model

