import numpy as np
import pickle
import pandas as pd
from acc_calc import accuracy
from q4 import NN
# from tensorflow.keras.models import load_model

STUDENT_NAME = 'Meng Tianao'
STUDENT_ID = '20810553'
trained_model = NN()
def test_mlp(data_file):
        # Load the test set
        # START
        dataset = pd.read_csv(data_file).to_numpy()
        # END

        # Load your network
        # START

        with open("trained_model.pkl", 'rb') as file:

            trained_model = pickle.loads(file.read())
        # END
        # Predict test set - one-hot encoded
        y_pred = trained_model.feedfoward_res(dataset)

        return y_pred


'''
How we will test your code:

from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 

y_pred = test_mlp('./test_data.csv')

test_labels = ...

test_accuracy = accuracy(test_labels, y_pred)*100
'''