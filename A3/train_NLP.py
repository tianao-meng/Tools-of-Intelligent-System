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

Epoch = 1
Batch_size = 16
documents = []
all_words = []

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


if __name__ == "__main__":
# 1. load your training data
    neg_train_file = glob.glob(r'data/aclImdb/train/neg/*.txt')
    pos_train_file = glob.glob(r'data/aclImdb/train/pos/*.txt')
    get_documents(neg_train_file,'neg',documents, all_words)
    get_documents(pos_train_file, 'pos', documents, all_words)

    random.shuffle(documents)

    all_words = nltk.FreqDist(all_words)
    word_features = list(all_words.keys())[:15000]

    word_features_model = open("models/word_features_model.pkl", 'wb')
    str = pickle.dumps(word_features)
    word_features_model.write(str)
    word_features_model.close()

    attribute_train, label_train = construct_feature_vector(word_features, documents)

  # build model for MLP
    model_MLP = Sequential()
    model_MLP.add(Flatten())
    model_MLP.add(Dense(512, activation=tf.nn.sigmoid))
    #model_MLP.add(Dense(512, activation=tf.nn.sigmoid))
    #model_MLP.add(Dense(512, activation=tf.nn.sigmoid))
    model_MLP.add(Dense(2, activation=tf.nn.sigmoid))
    model_MLP.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # predict = model_MLP.predict(attribute_test[0])



    # 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
    history_MLP = model_MLP.fit(attribute_train, label_train, epochs=Epoch, batch_size=Batch_size, validation_split=0.1)

    # Make sure you print the final training accuracy
    # training accuracy is shown in the process when you run, you can see
	# 3. Save your model
    NAME = f"{20810553}_NLP_model"
    model_MLP.save("models/{}".format(NAME))
    plt.show()
