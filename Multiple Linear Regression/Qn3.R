library(Hmisc)
coro <- read.csv(file.choose())
attach(coro)
summary(coro)

corof=coro[-c(1,2,5,6,8,10,11,12,15,19:38)]
#Find the correlation between Output & inputs - SCATTER DIAGRAM
pairs(corof)

#Correlation coefficient Matrix - Strength & Direction of correlation


cor(corof)

# The Linear Model of interest

model1<- lm(Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight)
summary(model1)


library(car)
vif(model1) # variance inflation factor

model2 <- lm(Price~Age_08_04+KM+HP+Gears+Quarterly_Tax+Weight)
summary(model2)

vif(model2)



n=nrow(corof)
n1=n*0.7
n2=n-n1
train=sample(1:n,n1)
test=corof[-train,]

pred=predict(model2,newdat=test)
actual=test$Price
error=actual-pred

test.rmse=sqrt(mean(error**2))
test.rmse

train.rmse = sqrt(mean(model2$residuals**2))
train.rmse


