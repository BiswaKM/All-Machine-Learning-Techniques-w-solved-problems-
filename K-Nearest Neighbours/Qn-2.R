library(class)
library(gmodels)
library(caret)
zoo<-read.csv('C:\\Users\\Biswa\\Desktop\\360DigiTMG\\18 . Module 18 - Machine Learning K-Nearest Neighbour\\Assignment\\Dataset\\zoo.csv')

colnames(zoo)
head(zoo)

#Copying our data for knn
data_class<-zoo

type<-zoo[,c(17:18)]
type$catsize<-NULL
data_class<-zoo[-18]
data_class$animal.name<-NULL

data_class=scale(data_class)
#Segregating our data into input & output sets
set.seed(1234)

smp_size=floor(0.75*nrow(data_class))
train_ind<-sample(nrow(data_class),size=smp_size)

#Creating test & train set
class_pred_train<-data_class[train_ind,]
class_pred_test<-data_class[-train_ind,]

type_out_train<-type[train_ind,]
type_out_test<-type[-train_ind,]

type_pred_knn<-knn(train=class_pred_train,test=class_pred_test,cl=type_out_train,k=10)

#Model Evaluation
type_out_test<-data.frame(type_out_test)

class_comparison<-data.frame(type_pred_knn,type_out_test)

names(class_comparison)<-c('Predicted Type','Observed Type')

head(class_comparison)

#Creating Table
CrossTable(x=class_comparison$`Observed Type`,y=class_comparison$`Predicted Type`,prop.chisq = FALSE,prop.c = FALSE,prop.r = FALSE,prop.t = FALSE)

#Using caret package for finding best k value
type_pred_caret<-train(class_pred_train,type_out_train,method = 'knn',preProcess = c('center','scale'))
type_pred_caret

plot(type_pred_caret)

#Confusion Martix

knnPredict<-predict(type_pred_caret,newdata = class_pred_test)

library(e1071)
confusionMatrix(factor(knnPredict,levels=1:7),factor(type_out_test$type_out_test,levels=1:7))

#Fianl Model wth K=9
type_pred_knn<-knn(train=class_pred_train,test=class_pred_test,cl=type_out_train,k=9)

#Model Evaluation
type_out_test<-data.frame(type_out_test)

class_comparison<-data.frame(type_pred_knn,type_out_test)

names(class_comparison)<-c('Predicted Type','Observed Type')

head(class_comparison)

#Creating Table
CrossTable(x=class_comparison$`Observed Type`,y=class_comparison$`Predicted Type`,prop.chisq = FALSE,prop.c = FALSE,prop.r = FALSE,prop.t = FALSE)

#Confusion Martix

knnPredict<-predict(type_pred_caret,newdata = class_pred_test)

library(e1071)
confusionMatrix(factor(knnPredict,levels=1:7),factor(type_out_test$type_out_test,levels=1:7))








