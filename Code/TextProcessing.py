import pandas as pd
import numpy as np

import nltk
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="white")

from nltk.corpus import stopwords

import tensorflow as tf

# Functions:
# remove punctuation and stopwords for word analysis:
stops = set(stopwords.words("english"))

def removePunctuation(text):
    puncts = ':#.?!,-\/()*&^%$_£;=[]{}+@<>\''
    for sym in puncts:
        text = text.replace(sym, ' ')
    return text

def removePunctuation_Sentence(text):
    puncts = ':#?!,-\/()*&^%$_£;=[]{}+@<>\'' # difference is no full stop to keep sentences.
    for sym in puncts:
        text = text.replace(sym, ' ')
    return text

def tidy_text(text):
    # punctuation:
    text = removePunctuation_Sentence(text)
    text = nltk.word_tokenize(text.lower()) # tokens
    # stop words:
    cleaned_text = ' '.join([word for word in text if word not in stops]) # remove stopwords     
    return cleaned_text

def count_infrequent_words(text, limit, doc_dict):    
    count = 0
    for t in text:
        if doc_dict[t] <= limit:
            count += 1
            
    return count

def count_clauses(text, clause_list):
    sents = nltk.sent_tokenize(text) # list of sentences in the paragraph: [sent[0], sent[1], .., sent[n-1]]
    list_clause_count = [sum([1 for c in clause_list if c in s]) for s in sents]
    return list_clause_count # output a list

def count_char_sentences(text):
    sents = nltk.sent_tokenize(text)
    char_len = [len(s) for s in sents]
    return char_len # return a list
    
def length_words_sentence(text):    
    sents = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(s) for s in sents]
    return [np.mean([len(w) for w in s]) for s in words]    
    
def create_summary_columns(val_list): # this function creates the summary columns where the input is a list
    return np.min(val_list), np.max(val_list), np.median(val_list), np.mean(val_list), np.percentile(val_list, 0.2), np.percentile(val_list, 0.8)
    
## Main text processing function - output dataframe with cleaned text column   
def create_processed_text(df):
    
    df['excerpt_lower'] = df.apply(lambda row : row['excerpt'].lower(), axis = 1)
    df['excerpt_letters'] = df.apply(lambda row : removePunctuation(row['excerpt_lower']), axis = 1)
    df['excerpt_cleaned'] = df.apply(lambda row : tidy_text(row['excerpt']), axis = 1)
    df['token_cleaned'] = df.apply(lambda row : row['excerpt_cleaned'].split( ), axis = 1)

    return df

def create_training_text(df, len_unique_words):
    # 1. one-hot encoding:
    df['embedded_sentences'] = df.apply(lambda row: tf.keras.preprocessing.text.one_hot(row['excerpt_cleaned'],
                                                        len_unique_words, split=' '), axis=1)
    
    # 2. Pad sentences so equal lengths:
    df['embedded_sentences_padded'] = tf.keras.preprocessing.sequence.pad_sequences(df['embedded_sentences']).tolist()

    return df    


## Function to create text-based features, e.g. average word length, average characters etc.
def create_text_features(df):

    # 2. Create Text Features:
    df['excerpt_lower'] = df.apply(lambda row : row['excerpt'].lower(), axis = 1)
    df['excerpt_letters'] = df.apply(lambda row : removePunctuation(row['excerpt_lower']), axis = 1)
    df['excerpt_cleaned'] = df.apply(lambda row : tidy_text(row['excerpt']), axis = 1)

    # dictionary of corpus and word count:
    corpus_freq = {}
    df['excerpt_tokens'] = df.apply(lambda row : nltk.word_tokenize(row['excerpt_letters']), axis = 1)
    for i in range(len(df.index)):
        for w in df['excerpt_tokens'].iloc[i]:
            if corpus_freq.get(w) == None:
                corpus_freq[w] = 1
            else:
                corpus_freq[w] += 1

    # count of excerpts containing a word - different to above because we only count word in an excerpt once
    document_freq = {} # so this adds up the total number of documents a word appears in
    # create a list of unique words in each string (cleaned excerpt)
    df['excerpt_cleaned_lower_tokens'] = df.apply(lambda row : list(set(row['excerpt_cleaned'].split(' '))), axis = 1)
    for i in range(len(df.index)):
        for w in df['excerpt_cleaned_lower_tokens'].iloc[i]:
            if document_freq.get(w) == None:
                document_freq[w] = 1
            else:
                document_freq[w] += 1
            
    doc_limit = 100
    df['infrequent_words'] = df.apply(lambda row : count_infrequent_words(row['excerpt_cleaned_lower_tokens'], doc_limit, document_freq), axis = 1)        

    df['word_count'] = df.apply(lambda row : len(row['excerpt_letters']), axis = 1)
    df['unique_words'] = df.apply(lambda row : len(set(row['excerpt_letters'])), axis = 1) 
    df['unique_word_ratio'] = df['unique_words'] / df['word_count']
    df['infrequent_word_ratio'] = df['infrequent_words'] / df['word_count']
    df[['target', 'word_count', 'unique_words', 'unique_word_ratio', 'infrequent_word_ratio', 'infrequent_words']].corr(method = 'spearman')

    clauses_list = ['and', 'but', 'so', 'therefore', 'because', ',', ';']
    df['clause_count'] = df.apply(lambda row : count_clauses(row['excerpt_lower'], clauses_list), axis = 1)
    df['char_count'] = df.apply(lambda row : count_char_sentences(row['excerpt_lower']), axis = 1)
    df['sentence_word_size'] = df.apply(lambda row : length_words_sentence(row['excerpt_lower']), axis = 1)
    df['clause_min'], df['clause_max'], df['clause_median'], df['clause_mean'], df['clause_20'], df['clause_80'] = zip(*df['clause_count'].map(create_summary_columns))
    df['char_min'], df['char_max'], df['char_median'], df['char_mean'], df['char_20'], df['char_80'] = zip(*df['char_count'].map(create_summary_columns))
    df['sentence_word_min'], df['sentence_word_max'], df['sentence_word_median'], df['sentence_word_mean'], df['sentence_word_20'], df['sentence_word_80'] = zip(*df['sentence_word_size'].map(create_summary_columns))

    return df

def embedding_dictionary(df):
    text_corpus = df['token_cleaned'].iloc[0]
    for i in range(1, len(df.index)):
        text_corpus += df['token_cleaned'].iloc[i]
    
    unique_words = list(set(text_corpus)) # list of words that appear in training and test dataset 
    len_unique_words = len(unique_words)

    word_dict = {} # pass this dictionry to the embedding function to assign vector representation to each unique word
    for word in unique_words:
        word_dict.update({word : tf.keras.preprocessing.text.one_hot(word, len_unique_words)})

    return word_dict, len_unique_words   

    
def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


