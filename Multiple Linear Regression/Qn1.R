library(Hmisc)
start <- read.csv(file.choose())
attach(start)
summary(start)

#Find the correlation between Output & inputs - SCATTER DIAGRAM
pairs(start)

#Correlation coefficient Matrix - Strength & Direction of correlation

start$State=as.numeric(start$State)
cor(start)

# The Linear Model of interest

model.start <- lm(Profit~R.D.Spend+Administration+Marketing.Spend+State)
summary(model.start)

library(car)
vif(model.start) # variance inflation factor

model2 <- lm(Profit~R.D.Spend+Administration+Marketing.Spend)
summary(model2)

n=nrow(start)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test=start[-train,]

pred=predict(model2,newdat=test)
actual=test$Profit
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse


