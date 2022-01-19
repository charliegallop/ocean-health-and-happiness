library(lme4)
library(lmerTest)
library(tidyverse)
library(magrittr)
library(sjPlot)
library(sjmisc)
library(sjstats)
library(arm)
library(readr)
df <- read_csv("/home/charlie/Documents/Uni/Exeter - Data Science/MTHM601_Fundamentals_of_Applied_Data_Science/assignmentProjectv2/data/cleanData/finalTable.csv")

lm1 <- lm(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec, data = df)
lmer1 <- lmer(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + (1|region), data = df)


summary(lmer1)

lm1 %>% se.coef() %>% round(5)
lmer1 %>% se.coef() %>% extract2(1) %>% round(5)

performance::icc(lmer1)
