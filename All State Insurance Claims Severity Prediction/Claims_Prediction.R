### Data Input Step ## 
train <- read.csv("D:/ML-in-R/ML-in-R/All State Insurance Claims Severity Prediction/Datasets/train.csv")
train<-train[1:2000,]


test <- read.csv("D:/ML-in-R/ML-in-R/All State Insurance Claims Severity Prediction/Datasets/test.csv")
test<-test[1:2000,]

head(train)

dim(train)

names(train)

# target variable is a continuous variable -it is a prediction problem

# exploratory data analysis
table(train$cat1);table(train$cat2);table(train$cat3);table(train$cat4);table(train$cat5)

table(train$cat6);table(train$cat7);table(train$cat8);table(train$cat9);table(train$cat10)

table(train$cat11);table(train$cat12);table(train$cat13);table(train$cat14);table(train$cat15)

df <-train[,c(2:117)]
apply(df,2,table)

plot(density(train$cont1))

# shapiro test
# Null : the distribution is normal
# Alt : the distribution is not normal


# shapiro test on cont1
shapiro.test(train[1:1000,"cont1"])
# we should not check the normality assumption for samples > 5000

plot(density(train$cont1))

# since the p-value is less than 0.05 we can rejectthe null hypothesis

par(mfrow=c(3,1))
plot(density(train$cont1));
plot(density(scale(train$cont1)));
plot(density(log(train$cont1)))

plot(density(log10(train$cont1)))

par(mfrow=c(3,1))
plot(density(train$cont1));plot(density(train$cont2));plot(density(train$cont3));plot(density(train$cont4))
plot(density(train$cont5));plot(density(train$cont6));plot(density(train$cont7));plot(density(train$cont8))
plot(density(train$cont9));plot(density(train$cont10));plot(density(train$cont11));plot(density(train$cont12))


par(mfrow=c(1,5))
boxplot(train$cont1);boxplot(train$cont2);boxplot(train$cont3);boxplot(train$cont4);boxplot(train$cont5)
boxplot(train$cont6);boxplot(train$cont7);boxplot(train$cont8);boxplot(train$cont9);boxplot(train$cont10)
boxplot(train$cont11);boxplot(train$cont12);boxplot(train$cont13);boxplot(train$cont14)

x7 <- boxplot(train$cont7)
length(x7$out)

x9 <- boxplot(train$cont9)
length(x9$out)

x10 <- boxplot(train$cont10)
length(x10$out)

dim(train)

# outlier definition
#XL < 1.5*IQR # outlier on the lower side
#XU > 1.5 * IQR # outlier on the higher side
#IQR = Q3-Q1

t<-quantile(train$cont7,prob=c(0.01,0.05,0.95,0.985));min(x7$out);max(x7$out);length(x7$out)

train$cont7_new<-ifelse(train$cont7>t[4],t[4],train$cont7)

boxplot(train$cont7_new);boxplot(train$cont7)


t<-quantile(train$cont9,prob=c(0.01,0.05,0.93,0.95))
t
train$cont9_new<-ifelse(train$cont9<t[1],t[1],ifelse(train$cont9>t[3],t[3],train$cont9))
par(mfrow=c(1,2))
boxplot(train$cont9);boxplot(train$cont9_new)

t<-quantile(train$cont10,prob=c(0.01,0.05,0.95,0.99))
t
#train$cont9_new<-ifelse(train$cont9<t[1],t[1],ifelse(train$cont9>t[3],t[3],train$cont9))#
#par(mfrow=c(1,2))
boxplot(train$cont10);#boxplot(train$cont9_new)

library(dplyr)
idVar = "id"
catVars = paste0("cat",seq(1,116))
contVars = paste0("cont",seq(1,14))
targetVar = "loss"


contVars

summary(train[,contVars])

corContPred = cor(train[,contVars])
corContPred

library(corrplot)

corrplot(corContPred,type="lower",order="hclust")

library(caret)
#check variance
zero.var = nearZeroVar(train, saveMetrics = T)
zero.var


dim(zero.var[zero.var$nzv == TRUE,])


head(zero.var[zero.var$nzv == TRUE,])

# regression experiment
fit<-lm(loss~ cont1+cont2+cont3+cont4+cont5+cont6+cont7+cont8+cont9+cont10+cont11+cont12+cont13+cont14+
          cat1+cat2+cat3+cat114+cat5+cat6+cat7+cat8+cat18,data=train)
summary(fit)

plot(density(fit$residuals))


fit1<-lm(loss~ cont1+cont2+cont3+cont4+cont5+cont6+cont7+cont8+cont9+cont10+cont11+cont12+cont13+cont14+
cat1+cat2+cat3+cat4+cat5+cat6+cat7+cat8+cat9+cat10+cat11+cat12+cat13+cat14+
#cat15+cat22+cat70+cat62+
cat16+cat17+cat18
+cat19+cat20+cat21+cat23+cat24+cat25+cat26+cat27+cat28+cat29+cat30+cat31+cat32+cat33+cat34+cat35
+cat36+cat37+cat38+cat39+cat40+cat41+cat42+cat43+cat44+cat45+cat46+cat47+cat48+cat49+cat50+cat51+cat52
#cat53+cat54+cat55+cat56+cat57+cat58+cat59+cat60+cat61+cat63+cat64+cat65+cat66+cat67+cat68+cat69+
#cat71+cat72+cat73+cat74+cat75+cat76+cat77+cat78+cat79+cat80+cat81+cat82+cat83+cat84+cat85+cat86+
#cat87+cat88+cat89+cat90+cat91+cat92+cat93+cat94+cat95+cat96+cat97+cat98+cat99+cat100+cat101+cat102+cat103+
#cat104+cat105+cat106+cat107+cat108+cat109+cat110+cat111+cat112+cat113+cat114+cat115+cat116
,data=train)
summary(fit1)

library(car)
library(MASS)

step_fit <- stepAIC(fit1,direction = "backward")

final_model <- lm(loss ~ cont1 + cont2 + cont4 + cont6 + cont7 + cont9 + cont10 + 
cont12 + cont14 + cat1 + cat6 + cat7 + cat8 + cat9 + cat10 + 
cat12 + cat13 + cat14 + cat16 + cat19 + cat20 + cat21 + cat23 + 
cat24 + cat25 + cat26 + cat27 + cat28 + cat34 + cat36 + cat37 + 
cat38 + cat40 + cat42 + cat43 + cat44 + cat46 + cat49 + cat51 + 
cat52,data=train)


summary(final_model)


# assumptions of a linear regression model
# residuals from the model follow a normal distribution
# no multicolinearity amongst the predictor variables
# no heteroscedasticity in the model
# no autocorrelation

vif(final_model) # always check if the vif for all the variables is less than 5




plot(final_model)


durbinWatsonTest(final_model)


durbinWatsonTest(fit1)

# libraries
library(readr)
library(data.table)
library(Matrix) # this is required for xgboost model
library(xgboost)

dim(train)

train$cont9<-train$cont9_new
train$cont9_new<-NULL


features = names(train)
features

dim(train);dim(test)


# XGBoost model
trainfeat<- data.table(train[-c(110,111,113,114,117)])
testfeat<- data.table(test[-c(110,111,113,114,117)])


testfact = cbind(testfeat,loss=0)


library(Matrix) # please install Matrix library - it is a must to run GBM
train_sparse_matrix = sparse.model.matrix(loss~.-1,data=trainfeat)
test_sparse_matrix = sparse.model.matrix(loss~.-1,data=testfact)

train_output_vector= trainfeat$loss

train_boost = xgboost(data=train_sparse_matrix,label=train_output_vector,
                      nrounds=1000,objective="reg:linear")
trainpreds = predict(train_boost,test_sparse_matrix)
testfact$loss = trainpreds


train_boost = xgboost(data=train_sparse_matrix,label=train_output_vector,
                      nrounds=1000,objective="reg:linear",
                      eta=0.3)
trainpreds = predict(train_boost,test_sparse_matrix)
testfact$loss = trainpreds
trainpreds


library(MLR) # hyper paramater ptimization model
library(parallelMap)


plot(trainpreds,train$loss,type="")
