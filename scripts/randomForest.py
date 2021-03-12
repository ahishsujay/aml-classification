import pandas as pd
import numpy as np
import pickle
import os, sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

#Argparse code:
parser = argparse.ArgumentParser()
parser.add_argument("-o", help="Output directory name entered in transformStats.py", required=True)
args = parser.parse_args()

#Populating variables:
output_path = args.o

#Transforming data for training and testing:
rf_dataset = pd.read_csv(output_path+"/inputData.csv", header=None)
rf_training = rf_dataset[rf_dataset[57].notnull()]
rf_training = rf_training.drop(columns=[56])
rf_training = rf_training.replace({57: {'normal':0, 'aml':1}})
rf_X = rf_training.iloc[:, :-1]
rf_y = rf_training.iloc[:, 56]

#Splitting the training and testing data:
rf_X_train, rf_X_test, rf_y_train, rf_y_test = train_test_split(rf_X, rf_y, test_size=0.25, random_state=5)

#Weights:
weight = np.array([6.78 if i == 0 else 1 for i in rf_y_train])

#Training:
rf = RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(rf_X_train, rf_y_train, sample_weight=weight)

#Predict:
rf_y_pred = rf.predict(rf_X_test)

#Evaluating the algorithm:
print(confusion_matrix(rf_y_test, rf_y_pred))
print(classification_report(rf_y_test, rf_y_pred))

#Score:
print("Accuracy: %.2f" %((metrics.accuracy_score(rf_y_test, rf_y_pred))*100))
print("F1 Score: %.2f" %(f1_score(rf_y_test, rf_y_pred)))

#Predicting on unlabelled data:
filename = 'randomForest_model.sav'
pickle.dump(rf, open(filename, 'wb'))

#Loading the randomForest model:
loaded_model = pickle.load(open(filename, 'rb'))

#Wrangling and transforming data to predict the labels:
dataset = pd.read_csv(output_path+"/inputData.csv", header=None)
dataset = dataset[dataset[57].isnull()]
dataset_patient = dataset.drop(columns=[57])
dataset = dataset.drop(columns=[56])
unlabelled_X = dataset.iloc[:, :-1]
unlabelled_X_array = unlabelled_X.to_numpy()
new_index = unlabelled_X.index.values.tolist()

#Predicting unlabelled data and saving final file:
unlabelled_y_array = loaded_model.predict(unlabelled_X_array)
unlabelled_y = pd.DataFrame(unlabelled_y_array)
unlabelled_y = unlabelled_y.rename(columns={0: 57})
unlabelled_y['id'] = unlabelled_y.index + 1790
unlabelled_y = unlabelled_y.set_index('id')
dataset_patient.merge(test, left_index=True, right_index=True).to_csv("predictedData.csv", header=None, index=None)

#Removing .sav file:
subprocess.run("rm randomForest_model.sav", shell=True)
