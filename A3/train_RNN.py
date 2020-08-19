# import required packages
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import preprocessing
Epochs=100

# commented part is the process of generating train_data_RNN.csv and test_data_RNN.csv
"""
np.set_printoptions(suppress=True)
# load q2_dataset.csv
def load_original_dataset():

    original_dataset_csv = np.loadtxt("data/q2_dataset.csv", delimiter=',', dtype=str)
    original_dataset_csv = list(original_dataset_csv)
    original_dataset_csv = original_dataset_csv[1:]

    processed_dataset = []
    for i in original_dataset_csv:
        processed_dataset.append(i[2:])
    #np.set_printoptions(suppress=True)
    processed_dataset = np.array(processed_dataset,dtype=float)

    return processed_dataset

def get_dataset(original_dataset):

    length = len(original_dataset)
    attribute = []
    label = []

    len_list = list(range(length))
    len_list_rever = len_list[::-1]
    #print("len_list_rever: ", len_list_rever)
    for i in len_list_rever:

        if ((i - 3) != -1):
            #print("i: ", i)
            attribute_ele = []
            #label_ele = []
            attribute_ele.append(original_dataset[i][0])
            attribute_ele.append(original_dataset[i][1])
            attribute_ele.append(original_dataset[i][2])
            attribute_ele.append(original_dataset[i][3])
            attribute_ele.append(original_dataset[i-1][0])
            attribute_ele.append(original_dataset[i-1][1])
            attribute_ele.append(original_dataset[i-1][2])
            attribute_ele.append(original_dataset[i-1][3])
            attribute_ele.append(original_dataset[i-2][0])
            attribute_ele.append(original_dataset[i-2][1])
            attribute_ele.append(original_dataset[i-2][2])
            attribute_ele.append(original_dataset[i-2][3])

            #label_ele.append(original_dataset[i+3][0])
            label.append(original_dataset[i-3][1])
            #label_ele.append(original_dataset[i+3][2])
            #label_ele.append(original_dataset[i+3][3])



            attribute.append(attribute_ele)
            #label.append(label_ele)
        else:
            break
    attribute = np.array(attribute)
    label = np.array(label)
    return attribute, label





def get_train_test_dataset(attribute, label, num_sample):


    total_index = list(range(num_sample))
    #print(total_index)
    random.shuffle(total_index)
    num_test = int(num_sample * 0.3)

    index_for_test = total_index[ : num_test]
    index_for_train = total_index[num_test : ]


    attribute_for_test = []
    label_for_test = []

    for i in index_for_test:
        attribute_for_test.append(attribute[i])
        label_for_test.append(label[i])

        for j in range(12):
            test_data_RNN_to_write = "{}{}".format(attribute[i][j],' ')
            test_data_RNN.write(test_data_RNN_to_write)
        

        test_data_RNN_to_write = "{}{}".format(label[i], '\n')
        test_data_RNN.write(test_data_RNN_to_write)




    attribute_for_train = []
    label_for_train = []

    for i in index_for_train:
        attribute_for_train.append(attribute[i])
        label_for_train.append(label[i])

        for j in range(12):

            train_data_RNN_to_write = "{}{}".format(attribute[i][j], ' ')
            train_data_RNN.write(train_data_RNN_to_write)
        
        train_data_RNN_to_write = "{}{}".format(label[i], '\n')
        train_data_RNN.write(train_data_RNN_to_write)



    attribute_for_train = np.array(attribute_for_train)
    label_for_train = np.array(label_for_train)
    attribute_for_test = np.array(attribute_for_test)
    label_for_test = np.array(label_for_test)
    return attribute_for_train, label_for_train, attribute_for_test, label_for_test
"""




# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
np.set_printoptions(suppress=True)
def load_train_dataset():

    train_dataset_csv = np.loadtxt("data/train_data_RNN.csv", delimiter=' ', dtype=str)

    train_dataset_csv = np.array(train_dataset_csv)

    return train_dataset_csv

def get_train_dataset(train_dataset):

    attribute = []
    label = []

    for i in train_dataset:
        attribute_ele = i[:12]
        label_ele = i[12]
        attribute.append(attribute_ele)
        label.append(label_ele)

    attribute = np.array(attribute,dtype=float)
    label = np.array(label,dtype=float)

    return attribute, label


if __name__ == "__main__":

    """
    original_dataset = load_original_dataset()

    train_data_RNN = open('data/train_data_RNN.csv', 'w')
    test_data_RNN = open('data/test_data_RNN.csv', 'w')
    
    attribute, label = get_dataset(original_dataset)
    print("attribute: ",attribute[0])
    print("label: ",label)
    num_sample = len(attribute)
    attribute_train, label_train, attribute_test, label_test = get_train_test_dataset(attribute, label, num_sample)
    print("label_train: ", len(label_train))
    print("label_test: ", len(label_test))
    """

    # 1. load your training data

    train_dataset = load_train_dataset()

    attribute, label = get_train_dataset(train_dataset)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    attribute = scaler.fit_transform(attribute)

    attribute = attribute.reshape(attribute.shape[0], 1, attribute.shape[1])

    # 2. Train your network
	# 		Make sure to print your training loss within training to show progress
    RNN_model = Sequential()
    RNN_model.add(LSTM(512, input_shape=(attribute.shape[1:]), return_sequences=True))

    #RNN_model.add(Dropout(0.2))
    #RNN_model.add(BatchNormalization())
    #RNN_model.add(LSTM(256, return_sequences=True))
    #RNN_model.add(LSTM(256, return_sequences=True))
    #RNN_model.add(Dropout(0.1))
    #RNN_model.add(BatchNormalization())
    #RNN_model.add(LSTM(64))
    RNN_model.add(LSTM(512))
    #RNN_model.add(Dropout(0.2))
    #RNN_model.add(BatchNormalization())


    RNN_model.add(Dense(512, activation='relu'))
    #RNN_model.add(Dense(256, activation='relu'))
    #RNN_model.add(Dense(256, activation='relu'))
    #RNN_model.add(Dropout(0.2))


    #opt = tf.keras.optimizers.Adam(lr=0.03, decay=1e-4)

    RNN_model.add(Dense(1))
    RNN_model.compile(loss='mae', optimizer='adam')
    history = RNN_model.fit(attribute, label, epochs=Epochs, validation_split=0.1)



    # Make sure you print the final training loss
    plt.figure(1)
    plt.title('train loss for RNN')
    x1 = range(Epochs)
    #plt.plot(x1, history.history['val_accuracy'], label = 'validation accuarcy', color = 'r')
    plt.plot(x1, history.history['loss'], label='train loss', color='b')
    plt.legend()
    plt.show()

    #3. Save your model
    NAME = f"{20810553}_RNN_model"
    RNN_model.save("models/{}".format(NAME))

