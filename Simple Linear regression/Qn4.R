slh<-read.csv(file.choose())
attach(slh)
plot(slh)
summary(slh)
cor(slh)
colnames(slh)
model1=lm(Salary~YearsExperience)
summary(model1)
predict(model1)
model1$residuals
confint(model1,level = 0.95)
predict(model1,interval = "confidence")
rmse1=sqrt(mean(model1$residuals^2))
rmse1

#log Model:
plot(log(YearsExperience),Salary)
cor(log(YearsExperience),Salary)
model2=lm(Salary~log(YearsExperience))
summary(model2)
rmse2=sqrt(mean(model2$residuals^2))
rmse2

#Exponential Model:
plot(YearsExperience,log(Salary))
cor(YearsExperience,log(Salary))
model3=lm(log(Salary)~YearsExperience)
summary(model3)
model3$residuals
log_salh=predict(model3,interval = "confidence")
log_salh
sh=exp(log_salh)
err1=Salary-sh
err1
rmse3=sqrt(mean(err1^2))
rmse3

#Polynomial Model:
model4=lm(log(Salary)~YearsExperience+I(YearsExperience*YearsExperience))
cor(YearsExperience+I(YearsExperience*YearsExperience),log(Salary))
summary(model4)
model4$residuals
exp_salh=predict(model4,interval = "confidence")
salh_poly=exp(exp_salh)
err_poly=Salary-salh_poly
err_poly
rmse4=sqrt(mean(err_poly^2))
rmse4
