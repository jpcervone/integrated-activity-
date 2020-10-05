# Prever a nota a ser dada por uma pessoa que ganha até 2 salários mínimos, por estado

library(tidyverse)
library(ISLR)
library(glmnet)
library(plotmo)
library(visdat)
library(rpart)
library(partykit)
library(skimr)
library(glmnet)
library(naniar)
library(rsample)
library(modeldata)
library(pROC)
library(randomForest)
library(MASS)


base_anatel <- read.csv2("C:/Users/chau_/OneDrive/Insper/1o T/Computação para CD/07. Atividade Integradora/BD_PRE_1.csv")

summary(base_anatel)

str(base_anatel)

class(base_anatel)

# selecionar as seguintes variáveis: renda, reclamação, número de operadoras

bf = base_anatel %>% select(J1, G1, C1_2, D2_1, D2_3, H2a)

bf_sna = na.omit(bf)

summary(bf)

nrow(bf_sna)

Y = ifelse(bf_sna$J1 >= 8, 1, 0)

bf_sna$J1 = NULL

X <- model.matrix(Y ~ ., data = bf_sna) 


# Criar conjunto de dados para treino

set.seed(888)

indice <- sample(nrow(bf_sna), size = 0.75*nrow(bf_sna), replace = FALSE) # sorteando os índices

indice


# modelo logístico

DF <- cbind(X, data.frame(Y = Y))
DF$J1 <- NULL

names(DF)

logistica <- glm(Y ~. -1, family = "binomial", data = DF[indice,])

y_logistica <- ifelse(predict(logistica, DF[-indice,], type = "response") >= 0.5, 1, 0)

mean(y_logistica != Y[-indice])

summary(logistica)

# Curva ROC
# lembrando: avalia conjuntamente sensibilidade e especificadade
# sensibilidade: probabilidade do modelo projetar positivo, quando o observado é positivo
# especificade: probabilidade do modelo projetar negativo, quando o observado é negativo

roc_fit <- roc(response = Y[-indice], predictor = y_logistica)

roc_fit

plot(roc_fit)


# Modelo Ridge

forecast_ridge <- glmnet(X[indice,], Y[indice], alpha = 0, nlambda = 500)

forecast_ridge

summary(forecast_ridge)

plot_glmnet(forecast_ridge, lwd = 2, cex.lab = 1.3)

vc_ridge <- cv.glmnet(X[indice,], Y[indice], alpha = 0)

coef(vc_ridge, s = vc_ridge$lambda.1se)

plot(vc_ridge, cex.lab = 1.3)

y_ridge <- predict(forecast_ridge, newx = X[-indice,], s = vc_ridge$lambda.1se)

summary(y_ridge)

erro_ridge <- mean((Y[-indice] - y_ridge)^2)

erro_ridge


# modelo Lasso

lasso <- glmnet(X[indice,], Y[indice], alpha = 1, nlambda = 1000)

summary(lasso)

plot_glmnet(lasso, lwd = 2, cex.lab = 1.3, xvar = "lambda")

vc_lasso <- cv.glmnet(X, Y, alpha = 1, 
                      lambda = lasso$lambda)

coef(vc_lasso, s = vc_lasso$lambda.1se)

plot(vc_lasso, cex.lab = 1.3)

y_lasso <- predict(lasso, newx = X[-indice,], s = vc_lasso$lambda.min)

erro_lasso <- mean((Y[-indice] - y_lasso)^2)

erro_lasso


# Floresta Aleatória

rf <- randomForest(as.factor(Y[indice]) ~ ., data = bf_sna[indice,], ntree = 800) # raiz de p preditoras

tibble(arvore = 1:nrow(rf$err.rate), mse = rf$err.rate[,1]) # o erro médio das 500 árvores


# mse out of bag ----------------------------------------------------------

tibble(arvore = 1:nrow(rf$err.rate), mse = rf$err.rate[,1]) %>% 
  ggplot(aes(x = arvore, y = mse)) + 
  geom_line(color = "#5B5FFF", size = 1.2) + 
  ylab("MSE (OOB)") + 
  xlab("Número de Árvores") + theme_bw()

# predict

mean((predict(rf, newdata = bf_sna[-indice,]) != as.factor(Y[-indice])))

# importância da variável -------------------------------------------------

varImpPlot(rf, pch = 19) # a fórmula da diferença com o Ypai
