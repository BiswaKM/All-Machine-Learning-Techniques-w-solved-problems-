dt<-read.csv(file.choose())
attach(dt)
summary(dt)

plot(Sorting.Time,Delivery.Time)
cor(Sorting.Time,Delivery.Time)
model1=lm(Delivery.Time~Sorting.Time)
model1
summary(model1)
predict(model1)
model1$residuals
confint(model1,level = 0.95)
predict(model1,interval = "confidence")
rmse1<-sqrt(mean(model1$residuals^2))
rmse1

#log Transformation:
plot(log(Sorting.Time),Delivery.Time)
cor(log(Sorting.Time),Delivery.Time)
model2=lm(Delivery.Time~log(Sorting.Time))
model2
summary(model2)
rmse2=sqrt(mean(model2$residuals^2))
rmse2

#Exponential Transformation:
plot(Sorting.Time,log(Delivery.Time))
cor(Sorting.Time,log(Delivery.Time))
model3=lm(log(Delivery.Time)~Sorting.Time)
summary(model3)
model3$residuals
log_dt1<-predict(model3,interval = "confidence")
log_dt1
at<-exp(log_dt1)
at
err1=Delivery.Time-at
err1
rmse3=sqrt(mean(err1^2))
rmse3

#Polynomial Transformation:
model4=lm(log(Delivery.Time)~Sorting.Time+I(Sorting.Time*Sorting.Time))
summary(model4)
confint(model4,level = 0.95)
log_dt<-predict(model4,interval = "confidence")
dt_poly<-exp(log_dt)
err_poly<-Delivery.Time-dt_poly
err_poly
rmse4<-sqrt(mean(err_poly))
rmse4

