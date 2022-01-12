#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 15:09:03 2022

@author: charlie
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

ref = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData/refugees.csv")
food = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData/foodInsecurity.csv")
un = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData/undernourishment.csv")

un['year'] = un['Year'].str.split('-', 1, expand = True)[1]
un.rename(columns = {'Country or Area':'country'}, inplace = True)
un['year'] = un['year'].astype(float)
un = un[un['year'] <2017]

ref2 = ref.iloc[:,[ 1, 2, 5]]
ref2.columns = ['country', 'year', 'value']
ref3 = ref2.groupby(by = ['country', 'year'],  as_index = False).sum()
ref4 = ref3[ref3['country'] == 'Afghanistan']


df = un.merge(ref3, how = 'left', on = ['country', 'year'])
df2016 = df[df['year'] == 2016]

x = df['Value']
y = df['value']

plt.scatter(x, y)