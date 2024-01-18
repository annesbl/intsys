import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pandas as pd

directory = "messdaten/"

file = "logfile_deo_dose_53mm.txt"

df = pd.read_csv(directory + file, header=None)





def label_data(df, label):
    df = df.drop(df.columns[0], axis=1)
    df['Label'] = label
    
    """ergänzt dataframe mit einer column, da steht welches objekt gemessen wurde
    """
    return df_labeled

labels = ['hallo']
df = label_data(df,labels)
print(df)