#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 20:07:40 2022

@author: charlie
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
sns.set_style('darkgrid')
from statsmodels.graphics.gofplots import ProbPlot
import statsmodels.api as sm


# %%

data = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/cleanData/cleanedDataImp.csv")
data2019 = data[data['year'] == 2019]


#%%

# Columns from 'data' for reference

# ['year',
#  'country',
#  'region',
#  'index',
#  'artisnalOpp',biodiversity+senseOfPlace+cleanWater+foodProv * region"
#  'biodiversity',
#  'carbonStorage',
#  'senseOfPlace',
#  'cleanWater',
#  'coastalProt',
#  'economies',
#  'foodProv',
#  'livelihoods',
#  'livelihoodsAndEconomics',
#  'natProducts',
#  'tourismAndRec',
#  'subFisheries',
#  'subHabitat',
#  'subIconicSp',
#  'subLastingSpecialPlaces',
#  'subMariculture',
#  'subSpCondition',
#  'happinessScore']
# %%
# %%


df = data[['year',
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
  'tourismAndRec','happinessScore']]
matrix = df.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.savefig("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/results/plots/Figure 2022-01-18 122732.png")


# %%
matrix = df.corr()
matrix = matrix.unstack()
matrix = matrix[abs(matrix) >= 0.3]

print(matrix)

#%%
import geopandas
# Getting geodataframe

data = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/cleanData/cleanedDataImp.csv")
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

modData = data.copy()
modData['countryMapMatch'] = modData['country']
modData.replace({"Ivory Coast":"Côte d'Ivoire", 
                 "Bosnia and Herzegovina":"Bosnia and Herz.",
                 "Dominican Republic":"Dominican Rep.",
                 "United States":"United States of America"}, inplace = True)
table = world.merge(modData, how="left", left_on=['name'], right_on=['countryMapMatch'], indicator = False)
table2 = table.dropna()
forExport = table2.drop('geometry', axis = 1)
forExport.to_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/cleanData/finalTable.csv")

#%%
df = forExport[[
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
  'tourismAndRec','happinessScore']]
matrix = df.corr().round(2)
sns.heatmap(matrix, annot=True)
plt.savefig("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/results/plots/Figure 2022-01-18 122732.png")

#%%
sns.pairplot(data,vars = ['senseOfPlace', 'cleanWater', 'tourismAndRec', 'happinessScore', 'coastalProt'], kind='reg')
# %%

# sns.lmplot(data = table2, x = 'cleanWater', y = 'happinessScore', col = 'year', hue = 'year', col_wrap = 3)

model1 = ols("happinessScore ~ cleanWater", data = table2).fit()
model2 = ols("happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + biodiversity + foodProv + natProducts", data = table2).fit()
model3 = ols("happinessScore ~ cleanWater + cleanWater * senseOfPlace", data = table2).fit()
model4 = ols("happinessScore ~ (cleanWater * senseOfPlace) + tourismAndRec + foodProv", data = table2).fit()

# plt.hist(model.resid)
print(model1.summary())
print(model2.summary())
print(model3.summary())
print(model4.summary())

# plt.clf()
# sns.histplot(model1.resid)
# plt.show()

# %%

# Diagnostic plot code from: https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/

def graph(formula, x_range, label=None):
    """
    Helper function for plotting cook's distance lines
    """
    x = x_range
    y = formula(x)
    plt.plot(x, y, label=label, lw=1, ls='--', color='red')


def diagnostic_plots(X, y, model_fit=None):
  """
  Function to reproduce the 4 base plots of an OLS model in R.

  ---
  Inputs:

  X: A numpy array or pandas dataframe of the features to use in building the linear regression model

  y: A numpy array or pandas series/dataframe of the target variable of the linear regression model

  model_fit [optional]: a statsmodel.api.OLS model after regressing y on X. If not provided, will be
                        generated from X, y
  """

  if not model_fit:
      model_fit = sm.OLS(y, sm.add_constant(X)).fit()

  # create dataframe from X, y for easier plot handling
  dataframe = pd.concat([X, y], axis=1)

  # model values
  model_fitted_y = model_fit.fittedvalues
  # model residuals
  model_residuals = model_fit.resid
  # normalized residuals
  model_norm_residuals = model_fit.get_influence().resid_studentized_internal
  # absolute squared normalized residuals
  model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
  # absolute residuals
  model_abs_resid = np.abs(model_residuals)
  # leverage, from statsmodels internals
  model_leverage = model_fit.get_influence().hat_matrix_diag
  # cook's distance, from statsmodels internals
  model_cooks = model_fit.get_influence().cooks_distance[0]

  plot_lm_1 = plt.figure()
  plot_lm_1.axes[0] = sns.residplot(model_fitted_y, dataframe.columns[-1], data=dataframe,
                            lowess=True,
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

  plot_lm_1.axes[0].set_title('Residuals vs Fitted')
  plot_lm_1.axes[0].set_xlabel('Fitted values')
  plot_lm_1.axes[0].set_ylabel('Residuals');

  # annotations
  abs_resid = model_abs_resid.sort_values(ascending=False)
  abs_resid_top_3 = abs_resid[:3]
  for i in abs_resid_top_3.index:
      plot_lm_1.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_residuals[i]));

  QQ = ProbPlot(model_norm_residuals)
  plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
  plot_lm_2.axes[0].set_title('Normal Q-Q')
  plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')
  plot_lm_2.axes[0].set_ylabel('Standardized Residuals');
  # annotations
  abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
  abs_norm_resid_top_3 = abs_norm_resid[:3]
  for r, i in enumerate(abs_norm_resid_top_3):
      plot_lm_2.axes[0].annotate(i,
                                 xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                     model_norm_residuals[i]));

  plot_lm_3 = plt.figure()
  plt.scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5);
  sns.regplot(model_fitted_y, model_norm_residuals_abs_sqrt,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
  plot_lm_3.axes[0].set_title('Scale-Location')
  plot_lm_3.axes[0].set_xlabel('Fitted values')
  plot_lm_3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

  # annotations
  abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
  abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
  for i in abs_norm_resid_top_3:
      plot_lm_3.axes[0].annotate(i,
                                 xy=(model_fitted_y[i],
                                     model_norm_residuals_abs_sqrt[i]));


  plot_lm_4 = plt.figure();
  plt.scatter(model_leverage, model_norm_residuals, alpha=0.5);
  sns.regplot(model_leverage, model_norm_residuals,
              scatter=False,
              ci=False,
              lowess=True,
              line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8});
  plot_lm_4.axes[0].set_xlim(0, max(model_leverage)+0.01)
  plot_lm_4.axes[0].set_ylim(-3, 5)
  plot_lm_4.axes[0].set_title('Residuals vs Leverage')
  plot_lm_4.axes[0].set_xlabel('Leverage')
  plot_lm_4.axes[0].set_ylabel('Standardized Residuals');

  # annotations
  leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
  for i in leverage_top_3:
      plot_lm_4.axes[0].annotate(i,
                                 xy=(model_leverage[i],
                                     model_norm_residuals[i]));

  p = len(model_fit.params) # number of model parameters
  graph(lambda x: np.sqrt((0.5 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50),
        'Cook\'s distance') # 0.5 line
  graph(lambda x: np.sqrt((1 * p * (1 - x)) / x),
        np.linspace(0.001, max(model_leverage), 50)) # 1 line
  plot_lm_4.legend(loc='upper right');


# %%

diagnostic_plots(table2[['senseOfPlace', 'cleanWater', 'tourismAndRec', 'artisnalOpp', 'biodiversity', 'foodProv', 'natProducts']], table2['happinessScore'])

# %%

diagnostic_plots(table2[['senseOfPlace']], table2['happinessScore'])

# %%

## Goodness of fit checks

# Fitted vs Residuals
plt.figure(figsize=(8,5))
p=plt.scatter(x=model2.fittedvalues,y=model2.resid,edgecolor='k')
xmin=min(model2.fittedvalues)
xmax = max(model2.fittedvalues)
plt.hlines(y=0,xmin=xmin*0.9,xmax=xmax*1.1,color='red',linestyle='--',lw=3)
plt.xlabel("Fitted values",fontsize=15)
plt.ylabel("Residuals",fontsize=15)
plt.title("Fitted vs. residuals plot",fontsize=18)
plt.grid(True)
plt.show()

#%%

# Normalised residuals

plt.figure(figsize=(8,5))
plt.hist(model2.resid_pearson,bins=20,edgecolor='k')
plt.ylabel('Count',fontsize=15)
plt.xlabel('Normalized residuals',fontsize=15)
plt.title("Histogram of normalized residuals",fontsize=18)
plt.show()

# %%

# QQplot

from statsmodels.graphics.gofplots import qqplot

plt.figure(figsize=(8,5))
fig=qqplot(model2.resid_pearson,line='45',fit='True')
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Theoretical quantiles",fontsize=15)
plt.ylabel("Sample quantiles",fontsize=15)
plt.title("Q-Q plot of normalized residuals",fontsize=18)
plt.grid(True)
plt.show()


# %%

# Normality (Shapiro-Wilk test)



from scipy.stats import shapiro

_,p=shapiro(model2.resid)


if p<0.01:
    print("The residuals seem to come from Gaussian process")
else:
    print("The normality assumption may not hold")


# %%

# Cook's Distance

from statsmodels.stats.outliers_influence import OLSInfluence as influence

inf=influence(model2)



(c, p) = inf.cooks_distance
plt.figure(figsize=(8,5))
plt.title("Cook's distance plot for the residuals",fontsize=16)
plt.stem(np.arange(len(c)), c, markerfmt=",")
plt.grid(True)
plt.show()

#%%








# ----------------------- TESTING DIFFERENT ANALYSIS-------------------------------
# %%
g = sns.FacetGrid(table2, col = 'region')
g.map_dataframe(sns.lmplot, "senseOfPlace", "happinessScore")

# %%

import statsmodels.api as sm
import statsmodels.formula.api as smf

md = smf.mixedlm("happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + coastalProt + biodiversity + carbonStorage + economies + foodProv + livelihoods + natProducts", table2, groups=table2['region'], re_formula='~year').fit()

print(md.summary())

# %%


from spreg import OLS
from spreg import MoranRes
from spreg import ML_Lag
from spreg import ML_Error
import pysal as ps

y = data['happinessScore']
X = data[['senseOfPlace', 'cleanWater']]
ols = OLS(y, X, name_y='home value', name_x=['income','crime'], name_ds='data', white_test=True)



#%%

# Spatial Weight Matrix


import libpysal
from libpysal  import weights
from libpysal.weights import Queen
import splot
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster, plot_local_autocorrelation
from splot.libpysal import plot_spatial_weights

import esda
from esda.moran import Moran, Moran_Local


W = weights.KNN.from_dataframe(table, k=6)
W.transform = 'R'
plot_spatial_weights(W, table);

# %%

# Spatial analysis regression
table2 = table.dropna()
y = table2['happinessScore'].values
y = y[np.logical_not(np.isnan(y))]
y_name = 'Happiness Score'

x = np.array([table2.senseOfPlace]).T
x = x[np.logical_not(np.isnan(x))]
x_name = 'Sense of Place'


ols = OLS(y = y, x = x, w = W,
          name_y=y_name, name_x = [x_name], name_w="W", name_ds='table2', 
          white_test=True, spat_diag=True, moran=True)
print(ols.summary)

# %%
from pysal.model import spreg

w = libpysal.weights.Rook.from_dataframe(table)

y = np.array(table2['happinessScore'])
y.shape = (len(table2['happinessScore']), 1)
variableNames = ['senseOfPlace', 'cleanWater', 'tourismAndRec', 'coastalProt']
x = []
for i in variableNames:
    x.append(table2[i])
x = np.array(x).T


m1 = spreg.OLS(y, x, name_y='happScore', name_x=variableNames)
#m2 = spreg.GM_Lag(y, x, w=W, name_y='happScore', name_x=variableNames)
print(m1.summary)
# %%

df_result = pd.DataFrame()
df_result['pvalues'] = model.pvalues[1:]


df_result['Features']=df_result.columns[:-1]



df_result.set_index('Features',inplace=True)

def yes_no(b):
    if b:
        return 'Yes'
    else:
        return 'No'
    


df_result['Statistically significant?']= df_result['pvalues'].apply(yes_no)



df_result



# %%


import warnings
warnings.filterwarnings('ignore')
sns.displot(data=data, x="cleanWater", kde=True, hue = "region")
sns.displot(data=data, x="senseOfPlace", y="happinessScore")
plt.show()



# %%
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
#%%
sns.lmplot(data = data, x = 'senseOfPlace', y = 'happinessScore')