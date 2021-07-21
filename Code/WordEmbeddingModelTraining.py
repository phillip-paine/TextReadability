import numpy as np 
import pandas as pd

import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")

from nltk.corpus import stopwords

import tensorflow as tf

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor  

def cnn_lstm_definition(embedding_layer):
    model_cnnlstm = tf.keras.Sequential() # sequential model to stack the layers of the neural net
    model_cnnlstm.add(embedding_layer) # add the first layer - the embedding layer.
    model_cnnlstm.add(tf.keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = "same", activation = "relu")) # 32 f
    model_cnnlstm.add(tf.keras.layers.Conv1D(filters = 8, kernel_size = 3, padding = "same", activation = "relu")) # 8 f   
    model_cnnlstm.add(tf.keras.layers.MaxPooling1D()) #maximum value in the feature map of each filter in the convolution layer    
    model_cnnlstm.add(tf.keras.layers.LSTM(20, activation='relu'))   
    model_cnnlstm.add(tf.keras.layers.Dense(30, activation = 'linear'))
    model_cnnlstm.add(tf.keras.layers.Dense(10, activation = 'linear'))    
    model_cnnlstm.add(tf.keras.layers.Dense(1)) # output layer

    #initial_learning_rate = 0.1
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate,
    #    decay_steps=100000,
    #    decay_rate=0.96,
    #    staircase=True)
    
    opti = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model_cnnlstm.compile(loss='mean_squared_error', optimizer=opti, metrics=['MeanSquaredError'])

    return model_cnnlstm

def cnn_lstm_cv(train_data, kfolds, test_data, p_epochs, embedding_matrix, embed_mat_dim):
    i = 1
    early_stopping_glove = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    for train_idx, test_idx in kfolds.split(train_data):    
        # 1. Model definition:
        embedding_layer = tf.keras.layers.Embedding(input_dim = embedding_matrix.shape[0], 
                            output_dim = embed_mat_dim, weights = [embedding_matrix], trainable = False)
        glove_model = cnn_lstm_definition(embedding_layer)

        # 2. Fit Model to each fold of training data:    
        train_split, val_split = train_data.iloc[train_idx, :], train_data.iloc[test_idx, :]
        train_target, val_target = train_data.iloc[train_idx, :]['centered_target'], train_data.iloc[test_idx, :]['centered_target']
        glove_model.fit(np.array(train_split['embedded_sentences_padded'].tolist()), train_target, callbacks = early_stopping_glove, 
                            epochs = p_epochs, validation_data= (np.array(val_split['embedded_sentences_padded'].tolist()), val_target), verbose = 0)   

        # 3. Predict output for training data - needed to fit the GBM to residuals:
        train_data.iloc[test_idx, :]['glove_predictions'] = glove_model.predict(np.array(val_split['embedded_sentences_padded'].to_list()))

        # 4. Create Output columns for test dataset    
        test_data['glove_predictions_model_' + str(i)] = glove_model.predict(np.array(test_data['embedded_sentences_padded'].to_list()))    
        i = i + 1

    test_data['glove_predictions'] = test_data[[col for col in test_data.columns if col.startswith('glove_predictions_model_')]].sum(axis=1) / 5 # average k-fold predictions

    return train_data, test_data

def Residual_GBM(train_data, test_data, feature_to_drop, feature_to_drop_test, gbm_params):

    train_target = train_data['glove_residual']

    train_features = train_data.drop(feature_to_drop, axis = 1)

    # Fit Model:    
    gbm_model_fitted = GradientBoostingRegressor(loss = gbm_params.get('loss'), learning_rate = gbm_params.get('learning_rate'), random_state=gbm_params.get('random_state'), 
                                                 n_estimators = gbm_params.get('n_estimators'), min_samples_split = gbm_params.get('min_samples_split'), 
                                                 min_samples_leaf = gbm_params.get('min_samples_leaf'), max_depth = gbm_params.get('max_depth'))
    gbm_model_fitted.fit(train_features, train_target)

    test_features = test_data.drop(feature_to_drop_test, axis = 1)

    # Create predictions in test dataset:L
    test_data['gbm_predictions'] = gbm_model_fitted.predict(test_features)

    return test_data


