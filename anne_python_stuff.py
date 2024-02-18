import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

"""skript liest messdaten aus dateien ein, transformiert und beschriftet die daten, 
   teilt sie in trainings und test sets auf, trainiert einen support vector classifier 
    und bewertet die genauigkeit"""

directory = "/Users/annesoballa/Documents/intsys übung/projekt/messungen/"

file = "logfile_deo_dose_53mm.txt"
file2 = "logfile_dose_zweite_messung.txt"
#file3 = "logfile_dose_dritte_messung.txt"
file4 = "logfile_rubicscube_1.txt"
file5 = "logfile_rubicscube_zweite_messung.txt"
df = pd.read_csv(directory + file , header=None)
df2 = pd.read_csv(directory + file2 , header=None)
df3 = pd.read_csv(directory + file4, header=None)
df4 = pd.read_csv(directory + file5 , header=None)

#transformiert  daten für klassifizierung
# 61 zeilen (eine Umdrehung)
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


df_new1 = transform_data(df)
df_new2 = transform_data(df2)
df_new3 = transform_data(df3)
df_new4 = transform_data(df4)
#df_new = pd.concat([df_new1, df_new2], axis=0)

"""datensätze mit labels versetzen und koombinieren
   label spalte erscheint zuerst"""

# spalte mit label für df_new1
df_new1['label'] = 'dose'
df_new2['label'] = 'dose'

# spalte mit label für df_new2
df_new3['label'] = 'rubicscube'
df_new4['label'] = 'rubicscube'


# beide df zusammenführen
df_new1 = pd.concat([df_new1, df_new2], axis=0)
df_new2 = pd.concat([df_new3, df_new4], axis=0)

df_new = pd.concat([df_new1, df_new2], axis=0)
# label spalte an erste stelle bringen
df_new = df_new[['label'] + [col for col in df_new.columns if col != 'label']]

print(df_new)



y = df_new['label']
X = df_new.drop('label', axis=1)


"""Daten werden in Trainings und testdaten aufgeteilt"""

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

"""support vector classifier und passt diesen den daten an"""
# clf = svm.SVC(gamma=0.001)
# clf.fit(X_train, y_train)

# """sagt labels für das testset vorher"""
# y_predicted = clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_predicted)
# print(f'Genauigkeit: {accuracy}')

# genauigkeit war bei 0,7



# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, y_predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")



# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# angles = np.linspace(0, 2 * np.pi, 61, endpoint=False)
# ax.plot(angles, df_new.iloc[:, 0])
# plt.show()

# import keras
# import tensorflow as tf
# from tensorflow.keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder

# # Datenvorbereitung für das neuronale Netzwerk
# X_train = X_train.values.reshape(-1, 61, 1)  # Umformen der Daten für das CNN
# X_test = X_test.values.reshape(-1, 61, 1)  # Umformen der Daten für das CNN

# # Label-Encoding für die Klassen
# label_encoder = LabelEncoder()
# y_train_encoded = label_encoder.fit_transform(y_train)
# y_test_encoded = label_encoder.transform(y_test)

# from keras.utils import to_categorical

# # One-Hot-Encoding für die Klassen
# y_train = to_categorical(y_train_encoded)
# y_test = to_categorical(y_test_encoded)


# from keras import models
# from keras import layers
# from keras import regularizers
# # Neuronales Netz mit verschiedenen Layern erstellen
# model = models.Sequential()
# model.add(layers.Conv1D(32, 3, activation = 'relu', input_shape = (61, 1)))
# model.add(layers.MaxPooling1D(2))
# model.add(layers.Conv1D(16, 3, activation = 'relu'))
# model.add(layers.MaxPooling1D(2))
# model.add(layers.Conv1D(16, 3, activation = 'relu'))
# model.add(layers.MaxPooling1D(2))
# model.add(layers.Dropout(0.4))
# model.add(layers.Flatten())
# model.add(layers.Dense(128, kernel_regularizer = regularizers.l2(0.007), activation='relu'))
# model.add(layers.Dense(64 , activation='relu'))
# model.add(layers.Dense(2, activation='softmax')) #anzahl der ausgabeneuronen
# # Kompilieren
# model.compile(
#     optimizer='rmsprop', 
#     loss='categorical_crossentropy', 
#     metrics=['accuracy']
# )
# # Zusammenfassung
# model.summary()


#trainieren

# import time

# #messen der trainingszeit
# start = time.time()

# #trainingsstart
# history = model.fit(X_train, y_train, epochs= 15, batch_size= 64, shuffle= True, validation_split= 0.2)
# print("Es hat:", time.time() - start )

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Ihre Daten X und Labels y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Ein-Klassen-zu-viele-Klassen-Transformation für die Labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Umwandlung der Labels in One-Hot-Encoding
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Erstellen des neuronalen Netzwerks
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')  # 3 Ausgabeneuronen für 3 Klassen
])

# Kompilieren des Modells
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training des Modells
history = model.fit(X_train, y_train_categorical, epochs=50, batch_size=32, validation_data=(X_test, y_test_categorical))

#plotten von verlust und genauigkeitskurve

# train_loss = history.history['loss']
# validation_loss = history.history['val_loss']
# train_accuracy = history.history['accuracy']
# validation_accuracy = history.history['val_accuracy']

# epochen = range(1, len(train_loss) +1)

# plt.figure(figsize=(12,4))


# #verlust plot
# plt.subplot(1, 2, 1)
# plt.plot(epochen, train_loss, 'b-', label= "Traininsverlust")
# plt.plot(epochen, validation_loss, 'r-', label= "Validationsverlust")
# plt.title('Trainings- und Validationsverlust')
# plt.xlabel('Epochen')
# plt.ylabel('Verlust')
# plt.legend()


# #genauigkeit plot
# plt.subplot(1,2,2)
# plt.plot(epochen, train_accuracy, 'b-', label= "Trainingsgenauigkeit")
# plt.plot(epochen, validation_accuracy, 'r-', label= "Validationsgenauigkeit")
# plt.title('Trainings- und Validationsgenauigkeit')
# plt.xlabel('Epochen')
# plt.ylabel('Genauigkeit')
# plt.legend()


# plt.tight_layout()
# plt.show()


# plotten von matrix

# import matplotlib.pyplot as plt
# import numpy as np

# # Umwandeln der Verlust- und Genauigkeitswerte in numpy-Arrays
# train_loss = np.array(history.history['loss'])
# validation_loss = np.array(history.history['val_loss'])
# train_accuracy = np.array(history.history['accuracy'])
# validation_accuracy = np.array(history.history['val_accuracy'])

# # Erstellen der Heatmap für die Verlustwerte
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(np.vstack([train_loss, validation_loss]), cmap='viridis', aspect='auto')
# plt.colorbar(label='Loss')
# plt.title('Trainings- und Validationsverlust')
# plt.xlabel('Epochen')
# plt.ylabel('Loss')

# # Erstellen der Heatmap für die Genauigkeitswerte
# plt.subplot(1, 2, 2)
# plt.imshow(np.vstack([train_accuracy, validation_accuracy]), cmap='viridis', aspect='auto')
# plt.colorbar(label='Accuracy')
# plt.title('Trainings- und Validationsgenauigkeit')
# plt.xlabel('Epochen')
# plt.ylabel('Accuracy')

# plt.tight_layout()
# plt.show()


#plotten von konfusionsmatrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Vorhersagen des Modells für die Testdaten
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_categorical, axis=1)

# Berechnung der Konfusionsmatrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Plotten der Konfusionsmatrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Konfusionsmatrix')
plt.xlabel('Vorhergesagte Klasse')
plt.ylabel('Tatsächliche Klasse')
plt.show()

"""# # Definiere X und y
# X = df_new.drop('label', axis=1)
# y = df_new['label']

# # Definiere dein Modell
# model = make_pipeline(StandardScaler(), SVC())

# # Führe Kreuzvalidierung durch
# cv_scores = cross_val_score(model, X, y, cv=5)  # Hier könntest du die Anzahl der Folds (cv) anpassen

# # Gib die Ergebnisse aus
# print("Kreuzvalidierungsgenauigkeit: ", cv_scores)
# print("Durchschnittliche Kreuzvalidierungsgenauigkeit: ", np.mean(cv_scores))

# # Plotten der Konfusionsmatrix
# # plt.figure(figsize=(8, 6))
# # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
# # plt.title('Konfusionsmatrix')
# # plt.xlabel('Vorhergesagte Klasse')
# # plt.ylabel('Tatsächliche Klasse')
# # plt.show()


# #GENAUIGKEIT
# #Kreuzvalidierungsgenauigkeit:  [0.73469388 0.85714286 0.73469388 0.75510204 0.75      ]
# #Durchschnittliche Kreuzvalidierungsgenauigkeit:  0.7663265306122449"""


"""Daten werden in Trainings und testdaten aufgeteilt"""

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

"""support vector classifier und passt diesen den daten an"""
# clf = svm.SVC(gamma=0.001)
# clf.fit(X_train, y_train)

# """sagt labels für das testset vorher"""
# y_predicted = clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_predicted)
# print(f'Genauigkeit: {accuracy}')

# genauigkeit war bei 0,7