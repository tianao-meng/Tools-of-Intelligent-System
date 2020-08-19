
from test_mlp import test_mlp, STUDENT_NAME, STUDENT_ID
from acc_calc import accuracy 
from q4 import NN
import pandas as pd
if __name__ == "__main__":
        y_pred = test_mlp('./test_data.csv')

        test_labels = pd.read_csv('./test_label.csv').to_numpy()

        test_accuracy = accuracy(test_labels, y_pred) * 100
        print("test_accuarcy: ", test_accuracy)