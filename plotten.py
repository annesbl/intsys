import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_dir = "/Users/annesoballa/Documents/intsys übung/projekt/messungen/"
m_1 = "logfile_deo_dose_53mm.txt"
m_2 = "logfile_rubicscube_1.txt"
m_3 = "logfile_rubicscube_zweite_messung.txt"
m_4 = "logfile_dose_zweite_messung.txt"
m_5 = "logfile_dose_dritte_messung.txt"

df_dose = pd.read_csv(data_dir + m_1, header=None)
df_cube = pd.read_csv(data_dir + m_2, header=None)
df_cube2 = pd.read_csv(data_dir + m_3, header=None)
df_dose2 = pd.read_csv(data_dir + m_4, header=None)
df_dose3 = pd.read_csv(data_dir + m_5, header=None)

# Daten einlesen
df_dose = pd.read_csv(data_dir + m_1, header=None)
df_cube = pd.read_csv(data_dir + m_2, header=None)
df_cube2 = pd.read_csv(data_dir + m_3, header=None)
df_dose2 = pd.read_csv(data_dir + m_4, header=None)
df_dose3 = pd.read_csv(data_dir + m_5, header=None)

#POLAR
# Subplots erstellen
fig, axs = plt.subplots(1, 5, figsize=(20, 5), subplot_kw=dict(polar=True))

# Für jede Datei einen Plot erstellen
axs[0].scatter(np.radians(df_dose.index * 6), df_dose[0], label='Dose 1', color='blue')
axs[1].scatter(np.radians(df_cube.index * 6), df_cube[0], label='Rubiks Cube 1', color='red')
axs[2].scatter(np.radians(df_cube2.index * 6), df_cube2[0], label='Rubiks Cube 2', color='green')
axs[3].scatter(np.radians(df_dose2.index * 6), df_dose2[0], label='Dose 2', color='orange')
axs[4].scatter(np.radians(df_dose3.index * 6), df_dose3[0], label='Dose 3', color='purple')

# Achsenbeschriftungen und Titel hinzufügen
for ax in axs:
    ax.set_xlabel('Index')
    ax.set_ylabel('Wert')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 20)
    ax.legend()
    ax.grid(True)

plt.suptitle('Scatter Plots der Messungen', fontsize=16)
plt.tight_layout()
plt.show()

#SCATTER

fig, axs = plt.subplots(1, 5, figsize=(20, 5))

# Für jede Datei einen Plot erstellen
axs[0].scatter(df_dose.index, df_dose[0], label='Dose 1', color='blue')
axs[1].scatter(df_cube.index, df_cube[0], label='Rubiks Cube 1', color='red')
axs[2].scatter(df_cube2.index, df_cube2[0], label='Rubiks Cube 2', color='green')
axs[3].scatter(df_dose2.index, df_dose2[0], label='Dose 2', color='orange')
axs[4].scatter(df_dose3.index, df_dose3[0], label='Dose 3', color='purple')

# Achsenbeschriftungen, Titel, und Limits hinzufügen
for ax in axs:
    ax.set_xlabel('Index')
    ax.set_ylabel('Wert')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 20)
    ax.legend()
    ax.grid(True)

plt.suptitle('Scatter Plots der Messungen', fontsize=16)
plt.tight_layout()
plt.show()