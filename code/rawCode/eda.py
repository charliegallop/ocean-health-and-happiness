#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:07:40 2022

@author: charlie
"""

import pandas as pd
import seaborn as sns

data = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/cleanData/cleanedData.csv")



sns.lmplot(data = combDf, x = 'Index', y = 'score', hue = 'year')