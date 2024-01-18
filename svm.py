# intsys
#intsys
import pandas as pd
from scipy.io import arff as sp
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt





data_dir = "/home/arabanus/Desktop/intlsys_logs/"
m_1 = "logfile_deo_dose_53mm.txt"
m_2 = "logfile_rubiks_cube.txt"
m_3 = "logfile_rubiks_cube.txt"
m_4 = "logfile_dose_zweite_messung.txt"
m_5 = "logfile_dose_dritte_messung.txt"

df_dose = pd.read_csv(data_dir + m_1, header=None)
df_cube = pd.read_csv(data_dir + m_2, header=None)
df_cube2 = pd.read_csv(data_dir + m_3, header=None)
df_dose2 = pd.read_csv(data_dir + m_4, header=None)
df_dose3 = pd.read_csv(data_dir + m_5, header=None)




# # PLOTTING

#     fig = plt.figure(figsize=(16, 9), dpi=70)

#     axis1 = fig.add_subplot(5, 1, 1)
#     axis1.plot(df_dose, color="r")
#     #axis1.plot(df_empty, color="b")
#     axis1.legend(["dose"])
#     for i in range(9):
#         axis1.axvline(x = 61 + i*61, color="g", label="rotation")

#     axis2 = fig.add_subplot(5, 1, 2)
#     axis2.plot(df_dose2, color="r")
#     #axis2.plot(df_empty, color="b")
#     axis2.legend(["dose 2"])
#     for i in range(9):
#         axis2.axvline(x = 61 + i*61, color="g", label="rotation")

#     axis3 = fig.add_subplot(5, 1, 3)
#     axis3.plot(df_dose3, color="r")
#     #axis3.plot(df_empty, color="b")
#     axis3.legend(["dose 3"])
#     for i in range(9):
#         axis3.axvline(x = 61 + i*61, color="g", label="rotation")

#     axis4 = fig.add_subplot(5, 1, 4)
#     axis4.plot(df_cube, color="r")
#     #axis4.plot(df_empty, color="b")
#     axis4.legend(["cube"])
#     for i in range(9):
#         axis4.axvline(x = 61 + i*61, color="g", label="rotation")

#     axis5 = fig.add_subplot(5, 1, 5)
#     axis5.plot(df_cube, color="r")
#     #axis5.plot(df_empty, color="b")
#     axis5.legend(["cube 2"])
#     for i in range(9):
#         axis5.axvline(x = 61 + i*61, color="g", label="rotation")


#     plt.show()

data_dir = "/home/arabanus/Desktop/intlsys_logs/"

m_1 = "logfile_deo_dose_53mm.txt"
df_dose = pd.read_csv(data_dir + m_1, header=None)

#1 steht für Dose 0 steht für nicht Dose
labels = [1] * len(df_dose)


X_train, X_test, y_train, y_test = train_test_split(df_dose, labels, test_size=0.5, shuffle=True)
#shuffle muss auf true gesetzt werden da datensatz anders aufgebaut ist als vorher, da das training alle sorten kennen muss. daher vorher mischen 'shuffle=true'

clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, y_train)

#vorhersage machen
predicted = clf.predict(X_test)


print(sum(predicted == y_test)/len(y_test))

#Anzahl der richtigen Vorhersagen im Verhältnis zur Gesamtzahl der Beispiele in Prozent.
# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("confusion Matrix")
# print(f"Confusion Matrix:\n{disp.confusion_matrix}")
# plt.show()

#pfad zum speichern von chat gpt
import joblib
joblib.dump(clf, '/Pfad/zum/Speicherort/deiner/Datei/svm_model.joblib')