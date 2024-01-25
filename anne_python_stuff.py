import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

directory = "/Users/annesoballa/Documents/intsys übung/projekt/messungen/"

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
    df_transposed = df
    output_data = []

    for i in range(244): # 244 because 305 - 61 (letzte Umdrehung wird rausgenommen, damit jeder neue teil eine komplette umdrehung hat)
        one_turn = df_transposed.iloc[i:i+61, :]
        # print(one_turn)
        output_data.append(one_turn)
    dfs_reset_index = [df.reset_index(drop=True) for df in output_data]
    result_df = pd.concat(dfs_reset_index, axis=1)
    return result_df


df_new = transform_data(df)




#df_new[0,'dose'] = 1 















print(df_new)
# print(df_new)


X = df_new['label']
y = df_new.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

clf = svm.SVC(gamma=0.001)

clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_predicted)
print(f'Genauigkeit: {accuracy}')

#* 100:.2f}%



# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# angles = np.linspace(0, 2 * np.pi, 61, endpoint=False)
# ax.plot(angles, df_new.iloc[:, 0])
# plt.show()

