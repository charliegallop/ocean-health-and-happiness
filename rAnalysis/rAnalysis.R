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

lm1 <- lm(happinessScore ~ cleanWater, data = df) #+ cleanWater + tourismAndRec, data = df)
lm2 <- lm(happinessScore ~ cleanWater * senseOfPlace, data = df)
lm3 <- lm(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + biodiversity + foodProv + natProducts, data = df)
lm4 <- lm(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + foodProv, data = df)

plot_model(lmer1, type = "re")

par(mfrow = c(2, 2))
plot(lm2)

lm3 <- lm(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + biodiversity + foodProv + natProducts, data = df)

ggplot(data = df, aes(x = index, y = happinessScore, col = region)) +
  geom_point() +
  stat_smooth(method = "lm", aes(col = region), alpha = 0.15) +
  xlab("Ocean Health Index") +
  ylab("Happiness Score") +
  ggtitle("Relationship between happiness scores and the ocean health index")


lmer1 <- lmer(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + biodiversity + foodProv + natProducts + (1|region), data = df)
summary(lmer1)

lmer2 <- lmer(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + biodiversity + foodProv + (1|region), data = df)
summary(lmer2)

lmer3 <- lmer(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + foodProv + (1|region), data = df)
summary(lmer3)

lmer4 <- lmer(happinessScore ~ senseOfPlace + cleanWater + artisnalOpp + foodProv + (1|region), data = df)
summary(lmer4)



lm4 %>% se.coef() %>% round(5)
lmer4 %>% se.coef() %>% extract2(1) %>% round(5)

performance::icc(lmer4)

lmer1vc <- VarCorr(lmer4)
print(lmer4vc, comp = "Variance")


lmerdiag <- data.frame(Residuals = resid(lmer4),
                       region = df$region,
                       Fitted = fitted(lmer4))
ggplot(data = lmerdiag, aes(x = Fitted, y = Residuals, col = region)) +
  geom_point() +
  facet_wrap(~region) +
  ggtitle("Lowest level residuals facetting by region")
