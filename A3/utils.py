# import required packages
import nltk
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pickle



def get_documents(FilePath, label, documents, all_words):
    for file_path in FilePath:
        documents_ele = []
        rev_ele = []
        f = open(file_path, "r")
        lines = f.readlines()
        
        for line in lines:
            line.strip()
            word_list = nltk.word_tokenize(line)
            for word in word_list:
                
                rev_ele.append(word.lower())
                all_words.append(word.lower())
    
        documents_ele.append(rev_ele)
        documents_ele.append(label)
        documents.append(documents_ele)
        
        
    f.close()



def construct_feature_vector(word_features,documents):
    attribute = []
    label = []
    for document in documents:
        feature_vectors = {}
        words = set(document[0])
        for word in word_features:
            if word in words:
                feature_vectors[word] = 1
            else:
                feature_vectors[word] = 0
        processed_data_ele = []
        vector = []
        for value in feature_vectors.values():
            vector.append(value)
        attribute.append(vector)
        label.append(document[1])
    for i in range(len(label)):
        if (label[i] == 'pos'):
            label[i] = 1
            continue
        else:
            label[i] = 0



    attribute = np.array(attribute, dtype = int)
    label = np.array(label,dtype = int)

    return  attribute, label




