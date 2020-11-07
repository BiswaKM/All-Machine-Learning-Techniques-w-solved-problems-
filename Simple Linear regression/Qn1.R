library(MASS)
library(faraway)
wgcc=read.csv(file.choose())
attach(wgcc)

#EDA:
hist(wgcc$Weight.gained..grams.)
boxplot(Weight.gained..grams.)
y<-boxplot(Weight.gained..grams.)
y$out
mean(Weight.gained..grams.)
median(Weight.gained..grams.)
hist(wgcc$Calories.Consumed)
boxplot(Calories.Consumed)
z<-boxplot(Calories.Consumed)
z$out
mean(Calories.Consumed)
median(Calories.Consumed)

#Modeling:
colnames(wgcc)
plot(Calories.Consumed,Weight.gained..grams.)
cor(Calories.Consumed,Weight.gained..grams.)
model1=lm(Weight.gained..grams.~Calories.Consumed)
model1
summary(model1)
predict(model1)
model1$residuals
confint(model1,level = 0.95)
predict(model1,interval = 'confidence')
rmse=sqrt(mean(model1$residuals^2))
rmse

#Log Model
plot(log(Calories.Consumed),Weight.gained..grams.)
cor(log(Calories.Consumed),Weight.gained..grams.)
model2=lm(Weight.gained..grams.~log(Calories.Consumed))
model2
summary(model2)
rmse2=sqrt(mean(model2$residuals^2))
rmse2

#Exponetial Model
plot(Calories.Consumed,log(Weight.gained..grams.))
cor(Calories.Consumed,log(Weight.gained..grams.))
model3=lm(log(Weight.gained..grams.)~Calories.Consumed)
summary(model3)
model3$residuals
log_cc=predict(model3,interval = 'confidence')
log_cc
cc=exp(log_cc)
err1=Weight.gained..grams.-cc
err1
rmse3=sqrt(mean(err1^2))
rmse3

#polynomial transformation
model4=lm(log(Weight.gained..grams.)~Calories.Consumed+I(Calories.Consumed*Calories.Consumed))
summary(model4)
confint(model4,level=0.95)    
log_pwg=predict(model4,interval = 'confidence')
log_pwg
wgpoly=exp(log_pwg)
err_poly=Weight.gained..grams.-wgpoly
err_poly
rmse4=sqrt(mean(err_poly^2))
rmse4
rmse3
rmse2
rmse

#Box-cox transformation
model5=boxcox(model1,lambda = seq(-3,3))
plot(model1)
best.lam=model5$x[which(model5$y==max(model5$y))]
model5.inv=lm((Weight.gained..grams.)^-1~Calories.Consumed)
plot(model5.inv)
summary(model5.inv)
summary(model1)