# aml-classification

## Overview:
The goal of the project is to predict AML or normal status of patients from flow cytometry data. The samples consist of 43 AML positive patients and 316 healthy donors. The samples were studied with flow cytometry to quantitate the expression of different protein markers. Since the flow cytometer could not measure all of the markers simultaneously, each patient’s sample was subdivided in 8 aliquots (“tubes”) and analyzed with different marker combinations, 5 markers per tube. Information for about half of the donors on whether they are healthy or AML positive is provided as training set. The challenge is to determine the state of health of the other half, based only on the provided flow cytometry data. The five markers for each tube correspond to different fluorophore-conjugated antibodies targeting specific proteins.

## Methodology:
All the features of the flow cytometry data (FS Lin, SS Log, FL1 Log, FL2 Log, FL3 Log, FL4 Log and FL5 Log) was retained for training and testing the model. Since  the  "FS  Lin"  column values  were  not  standardized, z-score standardization (x-average(x)/standard deviation(x)) was applied such that the mean was 0 and the standard deviation was 1. Since the standard deviation of the other columns were ≈ 0.1, multiplied the "FS Lin" by 0.1 to match the other columns. Next, I extracted the mean, median, standard deviation, skewness, kurtosis, interquartile range (0.25, 0.75), minimum and maximum values of the features. After this, I mean  normalized the resulting preprocessed data to obtain a consolidated dataset for 359 patients which was used for training and testing the model.<br/>

After the data was transformed and normalized, I used various machine learning models: Linear Regression, Logistic Regression, k-NN, SVM and Random Forest, to test which model would give me the best predictions. Random Forest provided to give the best result with an accuracy of 97.1%, F1 score of 84% and AUC-ROC of 0.88.<br/>

Used this trained Random Forest model to predict the AML/normal status of the remaining unlabelled data. Successfully predicted 19 of 20 AML patients and achieved an accuracy of 99.44%, F1 of 97.43% with 1 mistake (see attached picture for confusion matrix and results).

## Data availability:
1. Flow cytometry data for all 359 patients: http://pengqiu.gatech.edu/MLB/CSV.zip
2. Class labels: http://pengqiu.gatech.edu/MLB/AMLTraining.csv.zip

The data is from a DREAM challenge, and more information about it can be found here: https://www.synapse.org/#!Synapse:syn2887788/wiki/
