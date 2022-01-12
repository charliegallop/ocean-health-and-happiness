#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:25:40 2022

@author: charlie
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rootDir = "/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2"
rawDataDir = "/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData"
years = [2015, 2016, 2017, 2018, 2019]


rawData = pd.read_csv(f"{rawDataDir}/scores.csv")


df1 = rawData[rawData['dimension'] == 'score']
df1 = df1.drop(['goal', 'dimension', 'region_id'], axis = 1)

df2 = pd.pivot_table(df1, values = 'value', index = ['scenario', 'region_name'], columns = 'long_goal').reset_index()
ohiDf = df2.rename(columns = {'scenario':'year', 'region_name':'country'})
ohiDf = ohiDf[(ohiDf['year'] <2020) &(ohiDf['year'] >2014)]

columnsOrder = ['year', 'country', 'region', 'rank', 'score', 'gdpPerCap', 'family', 'lifeExpectancy', 'freedom', 'trust', 'generosity']

def tidyData(year):
    rawData = pd.read_csv(f"{rawDataDir}/{year}.csv")
    df = rawData.copy()
    df['year'] = year 
    
    if (year == 2018) or (year == 2019):
        df = df.rename(columns = {'Overall rank':'rank', 'Country or region':'country', 'Score':'score', 'GDP per capita':'gdpPerCap','Social support':'family', 'Healthy life expectancy': 'lifeExpectancy', 'Freedom to make life choices':'freedom', 'Generosity':'generosity', 'Perceptions of corruption':'trust'})
    
    if (year == 2015) or (year == 2016):
        
        if year == 2015:
            df = df.drop(['Standard Error', 'Dystopia Residual'], axis = 1)
        else:
            df = df.drop(['Lower Confidence Interval', 'Upper Confidence Interval', 'Dystopia Residual'], axis = 1)
        
        df = df.rename(columns = {'Country':'country', 'Region':'region','Happiness Rank':'rank', 'Happiness Score':'score', 'Economy (GDP per Capita)':'gdpPerCap','Family':'family', 'Health (Life Expectancy)': 'lifeExpectancy', 'Freedom':'freedom', 'Generosity':'generosity', 'Trust (Government Corruption)':'trust'})
    
    if (year == 2017):
        df = df.drop(['Whisker.high', 'Whisker.low', 'Dystopia.Residual'], axis = 1)
        df = df.rename(columns = {'Country':'country', 'Happiness.Rank':'rank', 'Happiness.Score':'score', 'Economy..GDP.per.Capita.':'gdpPerCap','Family':'family', 'Health..Life.Expectancy.': 'lifeExpectancy', 'Freedom':'freedom', 'Generosity':'generosity', 'Trust..Government.Corruption.':'trust'})
    
    return df

hapDfs = []

for year in years:
    hapDfs.append(tidyData(year))
    

regionDf = hapDfs[0][['country','region']].append(hapDfs[1][['country','region']])
regionsToAdd = [['Trinidad & Tobago', 'Latin America and Caribbean'],
                ['Northern Cyprus', 'Western Europe'],
                ['North Macedonia', 'Central and Eastern Europe'],
                ['Gambia', 'Sub-Sahara Africa'],
                ['Taiwan Province of China', 'Eastern Asia'],
                ['Hong Kong S.A.R., China', 'Eastern Asia']]
regionDf = regionDf.append(pd.DataFrame(regionsToAdd, columns=['country', 'region']), ignore_index = True)

regionDf = regionDf.drop_duplicates(subset=['country'])

    

for item in [2, 3, 4]:
    hapDfs[item] = hapDfs[item].merge(regionDf, how = "left", on = 'country')

# Check for missing
# missingRegions = []
# for item in hapDfs:
#     item = item[columnsOrder]
#     missingRegions.append(item[item['region'].isna()])

for count, item in enumerate(hapDfs):
    if count == 0:
        hapMerged = pd.DataFrame(item)
    else:
        hapMerged = hapMerged.append(item)

combDf = ohiDf.merge(hapMerged, how = "inner", on = ['year', 'country'])
combDf.to_csv(f"{rootDir}/data/cleanData/cleanedData.csv", index = False)



