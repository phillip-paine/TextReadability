import numpy as np 
import pandas as pd

import nltk

import tensorflow as tf
import tensorflow_hub as hub

import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold 

def sentence_encoder_model(train_data, test_data, kfolds, use_file_path):
    ## 5-fold cross-validation to predict the target score using cnn-lstm model:
    train_data['sent_embed_predictions'] = 0 # placeholder target value
    i = 1
    early_stopping_glovecnn = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    for train_idx, test_idx in kfolds.split(train_data):    
        # 1. Model definition:
        model = tf.keras.models.Sequential()
        model.add(hub.KerasLayer(use_file_path, input_shape=[],trainable=False,dtype=tf.string)) # don't include the .pb file in the input filepath, just the folder it is within
        model.add(tf.keras.layers.Dense(128))
        model.add(tf.keras.layers.Dense(64))
        model.add(tf.keras.layers.Dense(32))
        model.add(tf.keras.layers.Dense(1))

        model.compile(optimizer='adam',loss = "mean_squared_error")

        # 2. Fit Model to new fold: epoch = 250 needed 
        train_split, test_split = train_data.iloc[train_idx, :], train_data.iloc[test_idx, :]
        train_target, test_target = train_data.iloc[train_idx, :]['centered_target'], train_data.iloc[test_idx, :]['centered_target']
        model.fit(np.array(train_split['excerpt'].tolist()), train_target, callbacks = early_stopping_glovecnn, 
                            epochs = 50, validation_data= (np.array(test_split['excerpt'].tolist()), test_target), verbose = 0)
    
        # 3. Predict Output on training validation dataset in each fold:
        train_data.iloc[test_idx, :]['sent_embed_predictions'] = model.predict(np.array(test_split['excerpt'].to_list()))
    
        # 4. Create prediction for test dataset    
        test_data['sent_embed_predictions_model_' + str(i)] = model.predict(np.array(test_data['excerpt'].to_list()))    
        
        i += 1 # to create unique column names

    # Average test dataset prediction columns:
    test_data['sent_embed_predictions'] = test_data[[col for col in test_data.columns if col.startswith('sent_embed_predictions_model_')]].sum(axis=1) / 5

    return test_data, train_data
   




