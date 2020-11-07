library(Hmisc)
comp <- read.csv(file.choose())
attach(comp)
summary(comp)

#Find the correlation between Output & inputs - SCATTER DIAGRAM
pairs(comp)

#Correlation coefficient Matrix - Strength & Direction of correlation

comp$cd=as.numeric(comp$cd)
comp$multi=as.numeric(comp$multi)
comp$premium=as.numeric(comp$premium)
cor(comp)

# The Linear Model of interest

model1<- lm(price~speed+hd+ram+screen+cd+multi+premium+ads+trend)
summary(model1)


library(car)
vif(model1) # variance inflation factor

model2 <- lm(price~speed+hd+ram+screen+ads+trend)
summary(model2)

vif(model2)



n=nrow(comp)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test=comp[-train,]

pred=predict(model2,newdat=test)
actual=test$price
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse


