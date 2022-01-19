#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:25:40 2022

@author: charlie
"""

# %%

# Import packages

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.impute import KNNImputer

#%%

# Set directories

rootDir = "/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2"
rawDataDir = "/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData"

# %%

## Ocean Health Index tidying

# Reading in the Ocean Health Index data
# Source: https://oceanhealthindex.org/global-scores/data-download/

rawOHI = pd.read_csv(f"{rawDataDir}/scores.csv")

#%%


# The OHI dataset contains results for different dimensions of the assessment
# ('score', 'pressures', 'trend', 'resilience', 'status')  that contribute to 
# the overall score of each country. Since the 'score' dimension
# is the weighted average of all the otehr dimensions, I selected it for 
# analysis 

df1 = rawOHI[rawOHI['dimension'] == 'score']

# Dropping the columns that aren't necessary for the analysis. 'goal' and
# 'region_id' are shorthands for other columns and the dimension is 'score'
# so is also not needed
df1 = df1.drop(['goal', 
                'dimension', 
                'region_id'], 
               axis = 1)

# The raw data has each goal for each country as a new line so I pivoted the
# data to have only one observation for each country for a given year

df2 = pd.pivot_table(df1, values = 'value', index = ['scenario', 'region_name'], columns = 'long_goal').reset_index()

# Renamed the variables so they had consistent naming convention

cleanOHI = df2.rename(columns = {'scenario':'year', 
                                 'region_name':'country', 
                                 'Artisanal opportunities':'artisnalOpp',
                                 'Biodiversity':'biodiversity',
                                 'Carbon storage':'carbonStorage',
                                 'Clean water':'cleanWater',
                                 'Coastal protection':'coastalProt',
                                 'Economies':'economies',
                                 'Fisheries (subgoal)':'subFisheries',
                                 'Food provision':'foodProv',
                                 'Habitat (subgoal)':'subHabitat',
                                 'Iconic species (subgoal)':'subIconicSp',
                                 'Index':'index',
                                 'Lasting special places (subgoal)':'subLastingSpecialPlaces',
                                 'Livelihoods':'livelihoods',
                                 'Livelihoods & economies':'livelihoodsAndEconomics',
                                 'Mariculture (subgoal)':'subMariculture',
                                 'Natural products':'natProducts',
                                 'Sense of place':'senseOfPlace',
                                 'Species condition (subgoal)':'subSpCondition',
                                 'Tourism & recreation':'tourismAndRec'})
                                 
# cleanOHI = cleanOHI[(cleanOHI['year'] < 2020) & (cleanOHI['year'] > 2014)]

# %%

## World Happiness Report tidying

# Set variables

columnsOrder = ['year', 
                'country', 
                'region', 
                'rank', 
                'score', 
                'gdpPerCap', 
                'family', 
                'lifeExpectancy', 
                'freedom', 
                'trust', 
                'generosity']

years = [2015, 2016, 2017, 2018, 2019]

#%%

# The data for each year were a different csv file and each file had different
# naming conventions or columns. This function drops any unnecessary columns
# and standardises the variable names so they can all be merged into one
# dataset

def TidyData(year):
    rawData = pd.read_csv(f"{rawDataDir}/{year}.csv")
    df = rawData.copy()
    df['year'] = year 
    
    if (year == 2018) or (year == 2019):
        df = df.rename(columns = {'Overall rank':'rank', 
                                  'Country or region':'country', 
                                  'Score':'score', 
                                  'GDP per capita':'gdpPerCap',
                                  'Social support':'family', 
                                  'Healthy life expectancy': 'lifeExpectancy', 
                                  'Freedom to make life choices':'freedom', 
                                  'Generosity':'generosity', 
                                  'Perceptions of corruption':'trust'})
    
    elif (year == 2015) or (year == 2016):
        
        if year == 2015:
            df = df.drop(['Standard Error', 
                          'Dystopia Residual'], 
                         axis = 1)
        else:
            df = df.drop(['Lower Confidence Interval', 
                          'Upper Confidence Interval', 
                          'Dystopia Residual'], 
                         axis = 1)
        
        df = df.rename(columns = {'Country':'country', 
                                  'Region':'region',
                                  'Happiness Rank':'rank', 
                                  'Happiness Score':'score', 
                                  'Economy (GDP per Capita)':'gdpPerCap',
                                  'Family':'family', 
                                  'Health (Life Expectancy)': 'lifeExpectancy', 
                                  'Freedom':'freedom', 
                                  'Generosity':'generosity', 
                                  'Trust (Government Corruption)':'trust'})
    
    elif (year == 2017):
        df = df.drop(['Whisker.high', 
                      'Whisker.low', 
                      'Dystopia.Residual'], 
                     axis = 1)
        df = df.rename(columns = {'Country':'country', 
                                  'Happiness.Rank':'rank', 
                                  'Happiness.Score':'score', 
                                  'Economy..GDP.per.Capita.':'gdpPerCap',
                                  'Family':'family', 
                                  'Health..Life.Expectancy.': 'lifeExpectancy', 
                                  'Freedom':'freedom', 
                                  'Generosity':'generosity', 
                                  'Trust..Government.Corruption.':'trust'})

    return df

# %%

# Create an array where each element is the tidied version for each year

allYearsWHR = []

for year in years:
    allYearsWHR.append(TidyData(year))
    

# %%

# Since the 2017, 2018, and 2019 data doesn't have a column for 'region'
# I created a reference dataframe from the 2015 and 2016 datasets which
# contains all the countries with their relevant region

regionDf = allYearsWHR[0][['country','region']].append(allYearsWHR[1][['country','region']])

# %%

# Some countries did not have a region associated with them. Since there were
# only a few countries I assigned the region manually and added it on to the 
# regions database that will be merged to the merged WHR dataframe

regionsToAdd = [['Trinidad & Tobago', 'Latin America and Caribbean'],
                ['Northern Cyprus', 'Western Europe'],
                ['North Macedonia', 'Central and Eastern Europe'],
                ['Gambia', 'Sub-Sahara Africa'],
                ['Taiwan Province of China', 'Eastern Asia'],
                ['Hong Kong S.A.R., China', 'Eastern Asia']]

regionDf = regionDf.append(pd.DataFrame(regionsToAdd, columns=['country', 'region']), ignore_index = True)

regionDf = regionDf.drop_duplicates(subset=['country'])

# For the 2017, 2018, and 2019 dataset I used the regionDf reference df to
# assign the appropriate region to each country in these datasets.

for item in [2, 3, 4]:
    allYearsWHR[item] = allYearsWHR[item].merge(regionDf, how = "left", on = 'country')
    


# Check for missing regions
# missingRegions = []
# for item in hapDfs:
#     item = item[columnsOrder]
#     missingRegions.append(item[item['region'].isna()])


# %%

# Merge the tidied WHR datasets to form one master dataframe

for count, item in enumerate(allYearsWHR):
    if count == 0:
        mergedWHR = pd.DataFrame(item)
    else:
        mergedWHR = mergedWHR.append(item)
        
mergedWHR['region'].replace("Sub-Sahara Africa", "Sub-Saharan Africa", regex = True, inplace = True)
        
# %%

# Merge the WHR dataset witht the OHI dataset on country and year, keeping
# only the countries that have been assessed in both datasets since not all
# countries were assessed in the OHI data

combDf = cleanOHI.merge(mergedWHR, how = "inner", on = ['year', 'country'])

combDf = combDf[['year',
            'country',
            'region',
            'index',
            'artisnalOpp',
            'biodiversity',
            'carbonStorage',
            'senseOfPlace',
            'cleanWater',
            'coastalProt',
            'economies',
            'foodProv',
            'livelihoods',
            'livelihoodsAndEconomics',
            'natProducts',
            'tourismAndRec',
            'subFisheries',
            'subHabitat',
            'subIconicSp',
            'subLastingSpecialPlaces',
            'subMariculture',
            'subSpCondition',
            'score']]
combDf.rename(columns = {'score': 'happinessScore'}, inplace = True)

combDf.to_csv(f"{rootDir}/data/cleanData/cleanedData.csv", index = False)
nullDf = combDf[combDf.isnull().any(axis = 1)]
# %%

# Imputing missing data using iterative imputer

df = combDf.drop(columns = ['country', 'region'])
df.drop('region', axis = 1, inplace = True)
imputeIt = IterativeImputer()
imputedIt = imputeIt.fit_transform(df)
dfImputedIt = pd.DataFrame(imputedIt, columns = df.columns)

# %%

# Imputing missing data #2
df = combDf.drop(columns = ['country', 'region'], axis = 1)

imputeKNN = KNNImputer(n_neighbors = 2)
imputedKNN = imputeKNN.fit_transform(df)

dfImp = pd.DataFrame(imputedKNN, columns = ['year','index', 'artisnalOpp', 'biodiversity',
       'carbonStorage', 'senseOfPlace', 'cleanWater', 'coastalProt',
       'economies', 'foodProv', 'livelihoods', 'livelihoodsAndEconomics',
       'natProducts', 'tourismAndRec', 'subFisheries', 'subHabitat',
       'subIconicSp', 'subLastingSpecialPlaces', 'subMariculture',
       'subSpCondition', 'happinessScore'])

dfImp.insert(loc = 1, column = 'country', value = combDf['country'])
dfImp.insert(loc = 2, column = 'region', value = combDf['region'])
dfImp.to_csv(f"{rootDir}/data/cleanData/cleanedDataImp.csv", index = False)



