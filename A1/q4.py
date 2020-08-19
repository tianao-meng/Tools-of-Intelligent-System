import numpy as np
import random
import pickle

def load_train_dataset():
    # each element in attribute_txt is a attribute list representing a sample
    attribute_csv = np.loadtxt("//Users//mengtianao//Documents//ece657//a1//train_data.csv", delimiter=',', dtype=np.float)
    label_csv = np.loadtxt("//Users//mengtianao//Documents//ece657//a1//train_labels.csv", delimiter=',', dtype=np.float)
    attribute_csv = np.array(attribute_csv)
    label_csv = np.array(label_csv)

    return attribute_csv, label_csv

#devide thr original dataset to train dataset and test dataset ( 6 : 4 )
def div_train_test(attribute, label, sample_num):

    num_test = sample_num * 0.4

    attribute_test = []
    label_test = []

    attribute_train = []
    label_train = []

    index_total = np.array(range(sample_num))
    np.random.shuffle(index_total)

    for i in range(sample_num):
        if (i < num_test):
            attribute_test.append(attribute[index_total[i]])
            label_test.append(label[index_total[i]])
            continue

        attribute_train.append(attribute[index_total[i]])
        label_train.append(label[index_total[i]])

    attribute_train = np.array(attribute_train)
    label_train = np.array(label_train)
    attribute_test = np.array(attribute_test)
    label_test = np.array(label_test)
    return attribute_train, label_train, attribute_test, label_test

#define activation function
def activate_function(input):
    return 1 / (1 + np.exp(- input))

def activate_func_der(input):
    return np.dot(input, (1 - input).T)

class NN:
    def __init__(self):
        # default nodes in each layer
        self.inputSize = 784
        self.outputSize = 4
        self.hiddenSize = 392

        # random generate the weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # 784 * 392
        self.b1 = np.random.randn(self.hiddenSize)

        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # 392 * 4
        self.b2 = np.random.randn(self.outputSize)

    # input (1 * 784), W1 (784 * 392), W2(392 * 4)
    # input is train sample
    def feedfoward(self, input):

        self.hiddenlayer_input = np.dot(input, self.W1) + self.b1# 1 * 392

        self.hiddenlayer_output = activate_function(self.hiddenlayer_input)  #1 * 392

        self.outputlayer_input = np.dot(self.hiddenlayer_output, self.W2) + self.b2 # 1*4

        predict = activate_function(self.outputlayer_input)

        return predict

    def backpropagation(self, predict, output, input):

        # derivative error over output, when consider error = 1/2 ( output - predict) ^ 2  1 * 4
        delta_err_o2 = output - predict

        # derivative of activation function of output layer
        delta_output_layer = activate_func_der(predict)

        # derivative of output layer input over W2 1 * 392
        delta_i2_W2 = self.hiddenlayer_output

        #derivative of error over W2 4 * 392
        delta_err_W2 = np.dot(np.dot(delta_err_o2 , delta_output_layer).reshape(-1,1) ,delta_i2_W2.reshape(1,-1))

        delta_err_b2 = np.dot(delta_err_o2 , delta_output_layer)

        # derivative of output layer input over o1 392 * 4
        delta_i2_o1 = self.W2

        #derivative of o1 over i1
        delta_o1_i1 = activate_func_der(self.hiddenlayer_output)

        # derivative of i1 over W1 1 * 784
        delta_i1_W1 = input

        # derivative of error over W1 392 * 784
        delta_err_W1 = (np.dot(np.dot(np.dot(np.dot(delta_err_o2 , delta_output_layer).reshape(1,-1) ,delta_i2_o1.T) ,delta_o1_i1).T , delta_i1_W1.reshape(1, -1)))
        delta_err_b1 = np.dot(np.dot(np.dot(delta_err_o2 , delta_output_layer).reshape(1,-1) ,delta_i2_o1.T), delta_o1_i1)

        #update W2
        self.W2 += 0.03 * delta_err_W2.T
        for i in range(len(self.b2)):
            self.b2[i] += 0.03 * delta_err_b2[i]

        #update W1
        self.W1 += 0.03 * delta_err_W1.T
        for i in range(len(self.b1)):
            self.b1[i] += 0.03 * delta_err_b1[0][i]

    def feedfoward_res(self, test_attribute):
        res = []
        for i in range(len(test_attribute)):
            #print(i)
            predict = self.feedfoward(test_attribute[i])
            predict = np.rint(predict)
            res.append(predict)
        np.array(res)
        return res

    def train(self, input, output):
        predict = self.feedfoward(input)
        self.backpropagation(predict, output, input)


def check_output_same(predict, output):
    print("predict: ",predict)
    print("output: ", output)
    for i in range(4):
        if predict[i] != output[i]:
            return False

    return True

def train_classfier(attribute, label):

    sample_num = 0
    for i in label:
        sample_num += 1
    classfier = NN()

    correct_rate = 0
    for j in range(100):
        # get the train dataset and test dataset
        attribute_train, label_train, attribute_test, label_test = div_train_test(attribute, label, sample_num)
        #print("j: ",j)
        correct = 0
        if(correct_rate < 0.96):
            for i in range(len(attribute_train)):
                if ((i % 1000) == 0):
                    print(i)
                    #print("b1: ", classfier.W1)
                #print("input: ",attribute_train[i])
                classfier.train(attribute_train[i], label_train[i])
            for i in range(len(attribute_test)):
                #print(i)
                #print("attribute_test[i].shape", attribute_test[i].shape)
                predict = classfier.feedfoward(attribute_test[i])
                predict = np.rint(predict)

                if check_output_same(predict, label_test[i]):
                    correct += 1

            correct_rate = correct / len(label_test)
            print(correct_rate)
    return classfier


def test(classfier, test_attribute, test_label):
    res = []
    for i in range(len(test_attribute)):
        print(i)
        predict = classfier.feedfoward(test_attribute[i])
        predict = np.rint(predict)
        res.append(predict)

    res = np.array(res)
    count = 0
    for i in range(len(res)):
        if(check_output_same(res[i], label_test[i])):
            count += 1

    correct_rate = count / len(test_label)
    return correct_rate

if __name__ == "__main__":
    attribute, label = load_train_dataset()
    classfier = train_classfier(attribute, label)

    trained_model = open("trained_model.pkl", 'wb')
    str = pickle.dumps(classfier)
    trained_model.write(str)
    trained_model.close()

    attribute_train, label_train, attribute_test, label_test = div_train_test(attribute, label, len(label))
    correct_rate = test(classfier, attribute_test, label_test)

    print("correct_rate: ", correct_rate)





