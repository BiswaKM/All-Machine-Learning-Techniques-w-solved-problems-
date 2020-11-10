df=read.csv('C:\\Users\\Biswa\\Desktop\\360DigiTMG\\20 . Module 20 - Adaboost - Extreme Gradient Boosting\\Assignment\\Diabetes_RF.CSV')
library(caret)
df$Class.variable=as.numeric(df$Class.variable)
df$Class.variable=as.factor(df$Class.variable)
str(df)
#Accuracy with single model
inTraininglocal<-createDataPartition(df$Class.variable,p=.75,list = F)
training<-df[inTraininglocal,]
testing<-df[-inTraininglocal,]

library(C50)
model<-C5.0(training$Class.variable~.,data = training[,-9])
pred<-predict.C5.0(model,testing[,-9])
a<-table(testing$Class.variable,pred)

sum(diag(a))/sum(a)

########Bagging
acc<-c()
for(i in 1:11)
{
  inTraininglocal<-createDataPartition(df$Class.variable,p=.85,list = F)
  training1<-df[inTraininglocal,]
  testing<-df[-inTraininglocal,]
  fittree <- C5.0(training1$Class.variable~., data=training1[,-5])
  pred<-predict.C5.0(fittree,testing[,-9])
  a<-table(testing$Class.variable,pred)
  acc<-c(acc,sum(diag(a))/sum(a))
}
summary(acc)
mean(acc)

####################### Boosting
#Accuracy with single model with Boosting

inTraininglocal<-createDataPartition(df$Class.variable,p=.75,list = F)
training<-df[inTraininglocal,]
testing<-df[-inTraininglocal,]

model<-C5.0(training$Class.variable~.,data = training[,-9],trials=10)
pred<-predict.C5.0(model,testing[,-9])
a<-table(testing$Class.variable,pred)

sum(diag(a))/sum(a)

######## Bagging and Boosting
acc<-c()
for(i in 1:11)
{
  
  inTraininglocal<-createDataPartition(df$Class.variable,p=.85,list = F)
  training1<-df[inTraininglocal,]
  testing<-df[-inTraininglocal,]
  
  fittree <- C5.0(training1$Class.variable~., data=training1,trials=10)
  pred<-predict.C5.0(fittree,testing[,-9])
  a<-table(testing$Class.variable,pred)
  
  acc<-c(acc,sum(diag(a))/sum(a))
  
}

summary(acc)
mean(acc)
