import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pandas as pd

directory = "messdaten/"

file = "logfile_deo_dose_53mm.txt"

df = pd.read_csv(directory + file, header=None)

def transform_data(df):
    """
    input 305 measurement points, > 5 turns * 61 measurements
    für jeden messpunkt einer drehung die nächsten 61 punkte anhängen
         > ACHTUNG!: nur wenn es 61 oder mehr messpunkte bis zum ende des input df gibt!

         > d.h. es gibt 305 - 61 = 244 output zeilen im df
    """
    # transpose from (n, 1) -> (1, n)
    df_transposed = df.T
    output_data = []

    for i in range(244): # 244 because 305 - 61 (letzte Umdrehung wird rausgenommen, damit jeder neue teil eine komplette umdrehung hat)
        one_turn = df_transposed.iloc[:, i:i+61]
        print(one_turn)
        # output_data.append(one_turn)
     
    output_df = pd.DataFrame(output_data)

    return output_df



df_new = transform_data(df)

# print(df)
print(df_new)