df=read.csv('C:\\Users\\Biswa\\Desktop\\360DigiTMG\\20 . Module 20 - Adaboost - Extreme Gradient Boosting\\Assignment\\wbcd.CSV')
library(caret)
df$diagnosis=as.numeric(df$diagnosis)
df$diagnosis=as.factor(df$diagnosis)
df$id<-NULL
str(df)

#Accuracy with single model
inTraininglocal<-createDataPartition(df$diagnosis,p=.75,list = F)
training<-df[inTraininglocal,]
testing<-df[-inTraininglocal,]

library(C50)
model<-C5.0(training$diagnosis~.,data = training[,-1])
pred<-predict.C5.0(model,testing[,-1])
a<-table(testing$diagnosis,pred)

sum(diag(a))/sum(a)

########Bagging
acc<-c()
for(i in 1:11)
{
  inTraininglocal<-createDataPartition(df$diagnosis,p=.85,list = F)
  training1<-df[inTraininglocal,]
  testing<-df[-inTraininglocal,]
  fittree <- C5.0(training1$diagnosis~., data=training1[,-1])
  pred<-predict.C5.0(fittree,testing[,-1])
  a<-table(testing$diagnosis,pred)
  acc<-c(acc,sum(diag(a))/sum(a))
}
summary(acc)
mean(acc)

####################### Boosting
#Accuracy with single model with Boosting

inTraininglocal<-createDataPartition(df$diagnosis,p=.75,list = F)
training<-df[inTraininglocal,]
testing<-df[-inTraininglocal,]

model<-C5.0(training$diagnosis~.,data = training[,-1],trials=10)
pred<-predict.C5.0(model,testing[,-1])
a<-table(testing$diagnosis,pred)

sum(diag(a))/sum(a)

######## Bagging and Boosting
acc<-c()
for(i in 1:11)
{
  inTraininglocal<-createDataPartition(df$diagnosis,p=.85,list = F)
  training1<-df[inTraininglocal,]
  testing<-df[-inTraininglocal,]
  fittree <- C5.0(training1$diagnosis~., data=training1,trials=10)
  pred<-predict.C5.0(fittree,testing[,-1])
  a<-table(testing$diagnosis,pred)
  acc<-c(acc,sum(diag(a))/sum(a))
  
}

summary(acc)
mean(acc)
