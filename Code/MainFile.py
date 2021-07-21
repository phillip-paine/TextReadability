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
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor  

# custom functions:
import TextProcessing as tp
import WordEmbeddingModelTraining as mt
import UniversalSentenceEncoderTraining as ut
import ValidationFunctions as vf


def main_run(df_train, df_test, param_model):
    end_of_train = len(df_train.index) # to separate the train and test data again after computing corpus dictionary
    concat_dfs = [df_train, df_test]
    df_input = pd.concat(concat_dfs, ignore_index = True)

    # 1. Text processing step:
    df_cleaned = tp.create_processed_text(df_input)

    # 2. Word Embedding : 
    dict_embed, corpus_length = tp.embedding_dictionary(df_cleaned)    
    #embedding dim can be changed using the other txt files contained in Glove6B folder.
    embedding_matrix = tp.create_embedding_matrix(param_model.get('path_to_wordembed'), dict_embed, param_model.get('dim_embed'))

    # 3. Word Embedding Input:
    df_embedding_ready = tp.create_training_text(df_cleaned, corpus_length) # create padded embedding columns from cleaned text

    # 4. Create Text GBM Features:
    df_to_train = tp.create_text_features(df_embedding_ready)

    # 5. Create test dataset for evaluating model and create centered target using mean of training  
    df_train = df_to_train.iloc[:end_of_train, :] # training data
    df_test = df_to_train.iloc[end_of_train:, :] # test data
    
    train_target_mean = df_train['target'].mean() # add the target_mean to the final target prediction
    df_train['centered_target'] = df_train['target'] - train_target_mean        

    df_test['target_pred_baseline'] = train_target_mean # the baseline method uses the mean of target in the training dataset
    
    # Create Train-Validation 5-Fold Split for model fitting:
    cv_folds = KFold(n_splits = 5) # k-fold split to create training and validation datasets for CNN-LSTM with embedding layer (Sec. 5)

    # 5 i). Training CNN-LSTM with GLoVe Embedding Text Analysis Model:
    # 5-fold cross-validation to predict target in test dataset - average predictions for test dataset    
    df_train, df_test = mt.cnn_lstm_cv(df_train, cv_folds, df_test, param_model.get('n_epochs'), embedding_matrix, param_model.get('dim_embed')) 
    # contains training and prediction stages 
    df_train['glove_residual'] = df_train['centered_target'] - df_train['glove_predictions'] # create the residual column for the GBM
   
    # Training GBM for Residual with Feature Analysis Model:
    # Fit a GBM to the residual of the GLoVe Embedding + CNN-LSTM Predictions using text features such as number of characters, average word length etc. 
    drop_features = ['excerpt_lower', 'excerpt_letters', 'excerpt_cleaned', 'excerpt_cleaned_lower_tokens', 'glove_residual', 
                     'excerpt', 'target', 'centered_target', 'embedded_sentences', 'embedded_sentences_padded', 'token_cleaned', 'id', 'url_legal',
                     'license', 'excerpt_tokens', 'standard_error', 'clause_count', 'char_count', 'sentence_word_size']
    
    # test fetures to drop:
    drop_features_test = ['excerpt_lower', 'excerpt_letters', 'excerpt_cleaned', 'excerpt_cleaned_lower_tokens',
                          'excerpt', 'target', 'embedded_sentences', 'embedded_sentences_padded', 'token_cleaned', 'id', 'url_legal',
                          'license', 'excerpt_tokens', 'standard_error', 'clause_count', 'char_count', 'sentence_word_size', 'glove_predictions', 'target_pred_baseline',
                         'glove_predictions_model_1', 'glove_predictions_model_2', 'glove_predictions_model_3', 'glove_predictions_model_4', 'glove_predictions_model_5']
    df_test = mt.Residual_GBM(df_train, df_test, drop_features, drop_features_test, param_model) # outputs the test datasaet with gbm_predictions column

    # Predictions from the word-embedding method:
    df_test['target_pred_wordembedding'] = df_test['glove_predictions'] + df_test['gbm_predictions'] + train_target_mean

    # 5 ii) Training Universal Sentence Encoder Model:
    # Uses the prebuilt and trained universal sentence encoder instead of a word-level embedding model as in 5. i)
    df_test, df_train = ut.sentence_encoder_model(df_train, df_test, cv_folds, param_model.get('use_path'))

    # Predictions from the sentence-embedding method:
    df_test['target_pred_sentenceembedding'] = df_test['sent_embed_predictions'] + train_target_mean

    # 6. Validation Stage:    
    methods = ['target_pred_baseline', 'target_pred_wordembedding', 'target_pred_sentenceembedding'] # methods for the validation calculations (columns to compare)
    vf.evaluate_predictions(df_test, methods, param_model.get('plot_path')) # prints output of RMSE and writes plot to file - which file?

    return df_test

if __name__ == '__main__':
    embedmodel_params = {'dim_embed' : 50, 'n_epochs' : 150, 'path_to_wordembed' : 'D:/Phillip/GloVe_6B/glove.6B.50d.txt',
    'sent_embed_path' : 'D:/Phillip/universal-sentence-encoder-models/use/'}   
    gbm_params = {'loss' : 'ls', 'learning_rate' : 0.1, 'random_state' : 100, 'n_estimators' : 20,
                    'min_samples_split' : 30, 'min_samples_leaf' : 25, 'max_depth' : 4}
    other_params = {'set_seed' : 100, 'test_portion' : 0.15, 'plot_path' : 'D:/Phillip/GitHub/LitReadability_Local/Output/'}    
    data_path = 'D:/Phillip/GitHub/LitReadability_Local/Data'        
    data_split_run = True  
    out_path = 'D:/Phillip/GitHub/LitReadability_Local/Output/'
    all_params = {}
    all_params.update(embedmodel_params)
    all_params.update(gbm_params)
    all_params.update(other_params)

    if data_split_run:
        df_input = pd.read_csv(data_path + 'dataset.csv')        
        df_train, df_test = train_test_split(df_input, test_size = all_params.get('test_portion'), random_state = all_params.get('set_seed')) # random_state is seed for reproducibility
    else:        
        df_train = pd.read_csv(data_path + 'training.csv')
        df_test = pd.read_csv(data_path + 'testing.csv')       

    output_test = main_run(df_train, df_test, all_params)
    output_test.to_csv(out_path + 'testing_withpredictions.csv', index = False)

