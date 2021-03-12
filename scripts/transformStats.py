import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, subprocess, argparse

def transformStats(path, output):
    '''
    input: 1.path of csv files, 2. output dir
    output: transformed csv files
    '''
    output = output.rstrip("/")
    output = output + "/"
    csv_dir = sorted(os.listdir(path))

    i = 0
    file_count = 0
    df_final = []

    #label_data = pd.read_csv("/Users/ahishsujay/Documents/Georgia_Tech/ML_Bioscience_BMED_6517/AMLTraining.csv") Local path
    label_data = pd.read_csv("sampleData/AMLTraining.csv")
    label_list = label_data['Label'].tolist()
    samplenumber_list = label_data['SampleNumber'].tolist()
    samplenumber_label_dict = dict(zip(samplenumber_list, label_list))

    for file in csv_dir:
        df = pd.read_csv(path+file)
        df_describe = df.describe().drop('count') #Getting the stats and dropping count
        df_skew = df_describe.append((df.skew()).T, ignore_index=True) #Appending skew
        df_kurt = df_skew.append((df.kurt()).T, ignore_index=True) #Appending kurtosis
        #df_median = df_kurt.append((df.median()).T, ignore_index=True) #Appending median (Already present when doing describe, added by mistake)
        df_final.append(df_kurt) #Appending everything to df_final list
        i += 1

        #When it is the 8th tube file, stop appending to df_final and create file:
        if i == 8:
            file_count += 1
            df_writecsv = pd.concat(df_final, axis=1, ignore_index=True)
            df_writecsv['patient'] = file_count
            df_writecsv['label'] = samplenumber_label_dict[file_count]
            df_writecsv.to_csv(output+"transformStats_"+str(file_count)+".csv", index=False, header=False)
            i = 0 #Resetting so that it occurs for every 8th file
            df_final = [] #Reset

def normalizeDf(concat_file):
    entire_df = pd.read_csv(concat_file, header=None) #Reading file
    #entire_df = entire_df[entire_df[57].notnull()] #Removing null entries
    right_df = entire_df[entire_df.columns[-2:]] #Getting the patient and label cols
    left_df = entire_df.iloc[:, :-2] #Getting the transformed data

    #Normalizing:
    normalized_df = (left_df-left_df.mean())/left_df.std()

    #Merging back and saving (overwrites the inputData.csv file):
    normalized_df.merge(right_df, left_index=True, right_index=True).to_csv(concat_file, index=False, header=False)


def main():

    #Argparse code:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Input file directory path.", required=True)
    parser.add_argument("-o", help="Output directory name.", required=True)
    args = parser.parse_args()

    #Populating variables:
    input_path = args.i
    input_path = input_path.rstrip("/")
    input_path = input_path + "/"
    output_path = args.o

    #Creating directory:
    subprocess.run("mkdir "+str(output_path), shell=True)

    #Running functions:
    print("Executing transformStats:")
    transformStats(input_path, output_path)

    #Sort and concatenate file:
    print("Sorting and concatenating to produce inputData.csv:")
    subprocess.run("find "+str(output_path)+"/ -name 'transformStats_*' | sort -V | xargs cat > "+str(output_path)+"/inputData.csv", shell=True)

    #Normalizing:
    print("Normalizing:")
    concat_file = str(output_path)+"/inputData.csv" #Final file 'inputData.csv' can be found in the respective output directory
    normalizeDf(concat_file)

if __name__ == "__main__":
    main()
