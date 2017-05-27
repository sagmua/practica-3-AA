## ----setup, include=FALSE------------------------------------------------
set.seed(5)
knitr::opts_chunk$set(echo = TRUE)
setwd("~/")

library("AppliedPredictiveModeling")

library("caret")
library("leaps")
library("glmnet")


## ------------------------------------------------------------------------
datos_sudafrica = read.csv("./datos/africa.data")
head(datos_sudafrica)

## ------------------------------------------------------------------------
datos_sudafrica = datos_sudafrica[,-which(names(datos_sudafrica) == "row.names")]

## ------------------------------------------------------------------------
class(datos_sudafrica$famhist)

## ------------------------------------------------------------------------
#Cambiamos la columna 'famhist' que contiene caracteres por su equivalente en valores numéricos:
datos_sudafrica = data.frame(famhist = (ifelse(datos_sudafrica$famhist=="Absent",0,1)),
                        datos_sudafrica[,-which(names(datos_sudafrica) == "famhist")])

## ------------------------------------------------------------------------
#Si queremos obtener un conjunto de indices train para luego ejecutar un modelo lineal sobre el train:
train = sample (nrow(datos_sudafrica), round(nrow(datos_sudafrica)*0.8))
#definimos ambos conjuntos en dos data.frame diferentes:
sudafrica_train = datos_sudafrica[train,]
sudafrica_test = datos_sudafrica[-train,]

## ------------------------------------------------------------------------
color = c(rep('green',sum(datos_sudafrica$chd ==0)),rep('red',sum(datos_sudafrica$chd==1)))

pairs(datos_sudafrica, bg = color, pch = 22)

## ------------------------------------------------------------------------
sudafricaTrans = preProcess(datos_sudafrica, method = c("BoxCox", "center", "scale", "pca"),
                            thresh = 0.9)
sudafricaTrans$rotation

## ------------------------------------------------------------------------
nearZeroVar(sudafricaTrans$rotation)

## ------------------------------------------------------------------------
sudafricaTrans = preProcess(sudafrica_train[,-which(names(sudafrica_train) == "chd")],
                            method = c("BoxCox", "center", "scale"),thresh = 0.9)

sudafrica_train[,-which(names(sudafrica_train) == "chd")]=predict(sudafricaTrans,
         sudafrica_train[,-which(names(sudafrica_train) == "chd")])

sudafrica_test[,-which(names(sudafrica_test) == "chd")] =predict(sudafricaTrans,sudafrica_test[,-which(names(sudafrica_test) == "chd")])

## ------------------------------------------------------------------------
regsub_sudafrica =regsubsets(datos_sudafrica[,-which(names(datos_sudafrica) == "chd")],
                             datos_sudafrica[,which(names(datos_sudafrica) == "chd")])

summary(regsub_sudafrica)

## ------------------------------------------------------------------------
etiquetas = sudafrica_train[,which(names(sudafrica_train) == "chd")]
tr = sudafrica_train[,-which(names(sudafrica_train) == "chd")]
tr = as.matrix(tr)
crossvalidation =cv.glmnet(tr,etiquetas,alpha=0)
print(crossvalidation$lambda.min)

## ------------------------------------------------------------------------
modelo_reg = glmnet(tr,etiquetas,alpha=0,lambda=crossvalidation$lambda.min)
print(modelo_reg)

## ------------------------------------------------------------------------
modelo_reg = glmnet(tr,etiquetas,alpha=0,lambda=0)
print(modelo_reg)

## ------------------------------------------------------------------------
m_muestra_sudafrica = lm(chd ~ age, data=sudafrica_train)

## ----errorEtiquetas------------------------------------------------------
calculoErrorMatrizConfusion  = function (modelo, test, etiquetas, imprimir_matriz=TRUE){
  # Una vez calculado el modelo, empleamos la función predict
  # para obtener la probabilidad de cada etiqueta.
  prob_test = predict(modelo, test[,-which(names(test) == etiquetas)], type="response")

  pred_test = rep(0, length(prob_test))
   # predicciones por defecto 0
  pred_test[prob_test >=0.5] = 1
   # >= 0.5 clase 1
  matriz_conf = table(pred_test, test[,which(names(test) == etiquetas)])
  if (imprimir_matriz)
    print(matriz_conf)

  etest = mean(pred_test != test[,which(names(test) == etiquetas)])
}

## ------------------------------------------------------------------------
etest_mmuestrasud = calculoErrorMatrizConfusion(m_muestra_sudafrica, sudafrica_test, "chd")
etest_mmuestrasud

## ------------------------------------------------------------------------
m1_sudafrica = lm(chd ~ age + famhist, data=sudafrica_train)

etest_m1sud = calculoErrorMatrizConfusion(m1_sudafrica, sudafrica_test, "chd")
etest_m1sud

## ------------------------------------------------------------------------
m2_sudafrica = lm(chd ~ age + famhist + tobacco, data=sudafrica_train)

etest_m2sud = calculoErrorMatrizConfusion(m2_sudafrica, sudafrica_test, "chd")
etest_m2sud

## ------------------------------------------------------------------------
m3_sudafrica = lm(chd ~ age + famhist + tobacco + ldl, data=sudafrica_train)

etest_m3sud = calculoErrorMatrizConfusion(m3_sudafrica, sudafrica_test, "chd")
etest_m3sud

## ------------------------------------------------------------------------
m4_sudafrica = lm(chd ~ age + famhist + tobacco + ldl + typea, data=sudafrica_train)

etest_m4sud = calculoErrorMatrizConfusion(m4_sudafrica, sudafrica_test, "chd")
etest_m4sud

## ------------------------------------------------------------------------
m5_sudafrica = lm(chd ~ age + famhist + tobacco + ldl + typea + obesity, data=sudafrica_train)

etest_m5sud = calculoErrorMatrizConfusion(m5_sudafrica, sudafrica_test, "chd")
etest_m5sud

## ------------------------------------------------------------------------
m6_sudafrica = lm(chd ~ age + famhist + tobacco + ldl + typea + sbp, data=sudafrica_train)

etest_m6sud = calculoErrorMatrizConfusion(m6_sudafrica, sudafrica_test, "chd")
etest_m6sud

## ------------------------------------------------------------------------
m7_sudafrica = lm(chd ~ I(age^2) + famhist + tobacco + ldl + typea, data=sudafrica_train)

eout_m7sud = calculoErrorMatrizConfusion(m7_sudafrica, sudafrica_test, "chd")
eout_m7sud

## ------------------------------------------------------------------------
m8_sudafrica = lm(chd ~ I(age^3) + famhist + tobacco + ldl + typea, data=sudafrica_train)

eout_m8sud = calculoErrorMatrizConfusion(m8_sudafrica, sudafrica_test, "chd")
eout_m8sud

## ----generaErrorParticionSudafrica---------------------------------------
generarErrorParticionSudafrica = function(datos){
  #Si queremos obtener un conjunto de indices train para luego ejecutar un modelo lineal sobre el train:
  indices_train = sample (nrow(datos), round(nrow(datos)*0.7))
  #definimos ambos conjuntos en dos data.frame diferentes:
  sudafrica_train = datos[indices_train,]
  sudafrica_test = datos[-indices_train,]

  #TRANSFORMACIONES:
  sudafricaTrans = preProcess(sudafrica_train[,-which(names(datos_sudafrica) == "chd")], method = c("BoxCox", "center", "scale"),thresh = 0.8)
sudafrica_train[,-which(names(sudafrica_train) == "chd")] =predict(sudafricaTrans,sudafrica_train[,-which(names(sudafrica_train) == "chd")])
sudafrica_test[,-which(names(sudafrica_test) == "chd")] =predict(sudafricaTrans,sudafrica_test[,-which(names(sudafrica_test) == "chd")])

#EVALUACION DEL MODELO
modelo_sudafrica = lm(chd ~ I(age^2) + famhist + tobacco + ldl + typea, data=sudafrica_train)

etest = calculoErrorMatrizConfusion(modelo_sudafrica, sudafrica_test, "chd", FALSE)
etest
}

mean(replicate(100, generarErrorParticionSudafrica(datos_sudafrica)))

## ------------------------------------------------------------------------
datos_ozone = read.csv("./datos/LAozone.data")
head(datos_ozone)

## ------------------------------------------------------------------------
datos_ozone = datos_ozone[,-which(names(datos_ozone) == "doy")]

## ------------------------------------------------------------------------
#Si queremos obtener un conjunto de indices train para luego ejecutar un modelo lineal sobre el train:
train = sample (nrow(datos_ozone), round(nrow(datos_ozone)*0.7))
#definimos ambos conjuntos en dos data.frame diferentes:
ozone_train = datos_ozone[train,]
ozone_test = datos_ozone[-train,]

## ------------------------------------------------------------------------
pairs(datos_ozone, pch = 22)

## ------------------------------------------------------------------------
ozoneTrans = preProcess(datos_ozone, method = c("BoxCox", "center", "scale", "pca"),thresh = 0.9)
ozoneTrans$rotation

## ------------------------------------------------------------------------
nearZeroVar(ozoneTrans$rotation)

## ------------------------------------------------------------------------
ozoneTrans = preProcess(ozone_train[,-which(names(datos_ozone) == "ozone")], method = c("BoxCox", "center", "scale"),thresh = 0.9)
ozone_train[,-which(names(ozone_train) == "ozone")] =predict(ozoneTrans,ozone_train[,-which(names(ozone_train) == "ozone")])
ozone_test[,-which(names(ozone_test) == "ozone")] =predict(ozoneTrans,ozone_test[,-which(names(ozone_test) == "ozone")])

## ------------------------------------------------------------------------
regsub_ozone =regsubsets(datos_ozone[,-which(names(datos_ozone) == "ozone")], datos_ozone[,which(names(datos_ozone) == "ozone")])

summary(regsub_ozone)

## ------------------------------------------------------------------------
variable_respuesta = ozone_train[,which(names(ozone_train) == "ozone")]
tr = ozone_train[,-which(names(ozone_train) == "ozone")]
tr = as.matrix(tr)
crossvalidation =cv.glmnet(tr,variable_respuesta,alpha=0)
print(crossvalidation$lambda.min)

## ------------------------------------------------------------------------
modelo_reg = glmnet(tr,variable_respuesta,alpha=0,lambda=crossvalidation$lambda.min)
print(modelo_reg)

## ------------------------------------------------------------------------
modelo_reg = glmnet(tr,variable_respuesta,alpha=0,lambda=0)
print(modelo_reg)

## ----calculoErrorMedioIntervalo------------------------------------------

calculoErrorMedioIntervalo  = function (modelo, test, variable_respuesta){
  prob_test = predict(modelo, test[,-which(names(test) == variable_respuesta)])

  etest = mean(abs(prob_test - test[,which(names(test) == variable_respuesta)])/(max(ozone_test$ozone)-min(ozone_test$ozone)))

}

## ------------------------------------------------------------------------
m1_ozone = lm(ozone ~ temp, data = ozone_train)

## ------------------------------------------------------------------------
plot(ozone_test$temp, ozone_test$ozone)
abline(m1_ozone$coefficients)

## ------------------------------------------------------------------------
etest_m1ozone = calculoErrorMedioIntervalo(m1_ozone, ozone_test, "ozone")
etest_m1ozone

## ------------------------------------------------------------------------
m2_ozone = lm(ozone ~ temp + ibh, data = ozone_train)

etest_m2ozone = calculoErrorMedioIntervalo(m2_ozone, ozone_test, "ozone")
etest_m2ozone

## ------------------------------------------------------------------------
m3_ozone = lm(ozone ~ temp + ibh + humidity, data = ozone_train)

etest_m3ozone = calculoErrorMedioIntervalo(m3_ozone, ozone_test, "ozone")
etest_m3ozone

## ------------------------------------------------------------------------
m4_ozone = lm(ozone ~ temp + ibh + humidity + vis , data = ozone_train)

etest_m4ozone = calculoErrorMedioIntervalo(m4_ozone, ozone_test, "ozone")
etest_m4ozone

## ------------------------------------------------------------------------
m5_ozone = lm(ozone ~ temp + ibh + humidity + ibt, data = ozone_train)

etest_m5ozone = calculoErrorMedioIntervalo(m5_ozone, ozone_test, "ozone")
etest_m5ozone

## ------------------------------------------------------------------------
m6_ozone = lm(ozone ~ temp + ibh + humidity + vis + ibt, data = ozone_train)

etest_m6ozone = calculoErrorMedioIntervalo(m6_ozone, ozone_test, "ozone")
etest_m6ozone

## ------------------------------------------------------------------------
m7_ozone = lm(ozone ~ temp * ibh * humidity * ibt , data = ozone_train)

etest_m7ozone = calculoErrorMedioIntervalo(m7_ozone, ozone_test, "ozone")
etest_m7ozone

## ----generaErrorParticionOzone-------------------------------------------
generarErrorParticionOzone = function(datos){
  #Si queremos obtener un conjunto de indices train para luego
  #ejecutar un modelo lineal sobre el train:
  train = sample (nrow(datos), round(nrow(datos)*0.7))
  #definimos ambos conjuntos en dos data.frame diferentes:
  ozone_train = datos_ozone[train,]
  ozone_test = datos_ozone[-train,]

  #TRANSFORMACIONES:
  ozoneTrans = preProcess(ozone_train[,-which(names(datos_ozone) == "ozone")], method = c("BoxCox", "center", "scale"),thresh = 0.8)
ozone_train[,-which(names(ozone_train) == "ozone")] =predict(ozoneTrans,ozone_train[,-which(names(ozone_train) == "ozone")])
ozone_test[,-which(names(ozone_test) == "ozone")] =predict(ozoneTrans,ozone_test[,-which(names(ozone_test) == "ozone")])

#EVALUACION DEL MODELO
modelo_ozone = lm(ozone ~ temp * ibh * humidity * ibt, data = ozone_train)

etest = calculoErrorMedioIntervalo(modelo_ozone, ozone_test, "ozone")
etest
}

## ------------------------------------------------------------------------
mean(replicate(100, generarErrorParticionOzone(datos_ozone)))

