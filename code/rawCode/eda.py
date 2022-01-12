#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:07:40 2022

@author: charlie
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

data = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/cleanData/cleanedData.csv")
data2019 = data[data['year'] == 2019]

# sns.set_style("white")
# sns.set_palette("mako")
# sns.color_palette("mako")
# sns.lmplot(data = data, x = 'Sense of place', y = 'score', hue = 'year')
# plt.savefig("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/results/plots/plot.svg")

x = data2019['Index']
y = data2019['score']

sns.pairplot(data2019,vars = ['Biodiversity', 'Index', 'score', 'Sense of place'], kind='reg')

model = ols("y ~ Index + 'Sense of place'", data2019).fit()

print(model.summary())