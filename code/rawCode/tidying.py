
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

airPoll = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData/airPollution.csv")
suicideRate = pd.read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/rawData/suicideRates.csv")
suicideRate = suicideRate[suicideRate["Dim1"] == "Both sexes"]
suicideRate = suicideRate[['Location', 'Period', 'FactValueNumeric']].copy()

dfAP = airPoll[airPoll["Dim1"] == "Total"]
dfSR = suicideRate[(suicideRate["Period"] <=2016) & (suicideRate["Period"] >= 2010)]

df = dfAP.merge(dfSR, how = 'left', on = ["Location", "Period"])


sns.lmplot(x = 'FactValueNumeric_x',y = 'FactValueNumeric_y', data = df, hue = 'Location')
#plt.scatter(df['FactValueNumeric_x'], df['FactValueNumeric_y'], label= df['Location'])