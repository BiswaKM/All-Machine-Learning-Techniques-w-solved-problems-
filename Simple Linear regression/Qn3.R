emp<-read.csv(file.choose())
attach(emp)
summary(emp)
cor(emp)
plot(emp)
colnames(emp)
model1=lm(Salary_hike~Churn_out_rate)
summary(model1)
predict(model1)
model1$residuals
?residuals
confint(model1,level = 0.95)
predict(model1,interval = "confidence")
rmse1=sqrt(mean(model1$residuals^2))
rmse1

#Log Model
plot(log(Salary_hike),Churn_out_rate)
cor(log(Salary_hike),Churn_out_rate)
model2=lm(Churn_out_rate~log(Salary_hike))
summary(model2)
rmse2=sqrt(mean(model2$residuals^2))
rmse2

#Exponential Model
plot(Salary_hike,log(Churn_out_rate))
cor(Salary_hike,log(Churn_out_rate))
model3=lm(log(Churn_out_rate)~Salary_hike)
summary(model3)
model3$residuals
log_emp=predict(model3,interval = "confidence")
log_emp
em=exp(log_emp)
err1=Churn_out_rate-em
err1
rmse3=sqrt(mean(err1^2))
rmse3

#polynomial Model
model4=lm(log(Churn_out_rate)~Salary_hike+I(Salary_hike*Salary_hike))
summary(model4)
confint(model3,level = 0.95)
log_em1=predict(model4,interval = "confidence")
em1_poly=exp(log_em1)
err_poly=Churn_out_rate-em1_poly
err_poly
rmse4=sqrt(mean(err_poly^2))
rmse4
