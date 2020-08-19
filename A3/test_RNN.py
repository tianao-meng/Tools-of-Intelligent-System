# import required packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Bidirectional, LSTM
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import preprocessing
from tensorflow.keras.models import load_model

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

np.set_printoptions(suppress=True)
def load_test_dataset():

    test_dataset_csv = np.loadtxt("data/test_data_RNN.csv", delimiter=' ', dtype=str)

    test_dataset_csv = np.array(test_dataset_csv, dtype=float)

    return test_dataset_csv

def get_test_dataset(test_dataset):

    attribute = []
    label = []

    for i in test_dataset:
        attribute_ele = i[:12]
        label_ele = i[12]
        attribute.append(attribute_ele)
        label.append(label_ele)

    attribute = np.array(attribute,dtype=float)
    label = np.array(label,dtype=float)

    return attribute, label


if __name__ == "__main__":
    # 1. Load your saved model
    RNN_model = load_model('models/20810553_RNN_model')

    # 2. Load your testing data
    test_dataset = load_test_dataset()

    test_dataset = test_dataset[np.lexsort(-test_dataset.T)]

    attribute, label = get_test_dataset(test_dataset)

    #preprocessing
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    attribute = scaler.fit_transform(attribute)

    attribute = attribute.reshape(attribute.shape[0], 1, attribute.shape[1])


    # 3. Run prediction on the test data and output required plot and loss
    predicted_res = RNN_model.predict(attribute)

    test_loss_RNN = RNN_model.evaluate(attribute, label)
    print("test_loss_RNN: ", test_loss_RNN)

    pre_res = []
    for i in predicted_res:
        pre_res.append(i[0])

    plt.figure(1)
    plt.title('Pre and True')
    plt.plot(list(range(len(label))), label, label = 'true', color = 'r')
    plt.plot(list(range(len(label))),pre_res, label='predict', color='b')
    plt.legend()
    plt.show()


