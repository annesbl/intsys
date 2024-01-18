#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:17:38 2023

@author: arabanus
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 101)

y = 4 + 2 * np.sin(2 * x)
z = 4 + 2 * np.cos(2 * x)


fig, ax = plt.subplots()

ax.plot(x, y, linewidth=2.0, color='blue')
ax.plot(x, z, linewidth=2.0, color='red')

ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
       ylim=(0, 8), yticks=np.arange(1, 8))

plt.show()