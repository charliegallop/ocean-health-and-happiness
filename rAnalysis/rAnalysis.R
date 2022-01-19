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

lm1 <- lm(happinessScore ~ senseOfPlace, data = df) #+ cleanWater + tourismAndRec, data = df)
lmer1 <- lmer(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + (1|region), data = df)
lm2 <- lm(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + biodiversity + foodProv + natProducts, data = df)
lmer2 <- lmer(happinessScore ~ senseOfPlace + cleanWater + tourismAndRec + artisnalOpp + biodiversity + foodProv + natProducts + (1|region), data = df)
summary(lm2)

lm1 %>% se.coef() %>% round(5)
lmer1 %>% se.coef() %>% extract2(1) %>% round(5)

performance::icc(lmer1)

lmer1vc <- VarCorr(lmer1)
print(lmer1vc, comp = "Variance")


lmerdiag <- data.frame(Residuals = resid(lmer1),
                          region = df$region,
                          Fitted = fitted(lmer1))
ggplot(data = lmerdiag, aes(x = Fitted, y = Residuals, col = region)) +
  geom_point() +
  facet_wrap(~region) +
  ggtitle("Lowest level residuals facetting by region")

plot_model(lmer1, type = "re")

par(mfrow = c(2, 2))
plot(lm2)
