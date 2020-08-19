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
from utils import get_documents, construct_feature_vector
from tensorflow.keras.models import load_model

documents = []
all_words = []
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow




if __name__ == "__main__": 
    # 1. Load your saved model
    with open("models/word_features_model.pkl", 'rb') as word_features_model:
        word_features = pickle.loads(word_features_model.read())

    NLP_model = load_model('models/20810553_NLP_model')

    # 2. Load your testing data
    neg_test_file = glob.glob(r'data/aclImdb/test/neg/*.txt')
    pos_test_file = glob.glob(r'data/aclImdb/test/pos/*.txt')
    get_documents(neg_test_file, 'neg', documents, all_words)
    get_documents(pos_test_file, 'pos', documents, all_words)

    attribute_test, label_test = construct_feature_vector(word_features, documents)

    # 3. Run prediction on the test data and print the test accuracy
    test_loss_NLP_model, test_accuarcy_NLP_model = NLP_model.evaluate(attribute_test, label_test)
    print("NLP model accuarcy: ", test_accuarcy_NLP_model)


