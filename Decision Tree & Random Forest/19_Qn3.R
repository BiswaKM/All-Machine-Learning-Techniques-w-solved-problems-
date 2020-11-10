fraud=read.csv('C:\\Users\\Biswa\\Desktop\\360DigiTMG\\19 . Module 19 - Decision Tree - Random Forest\\Assignment\\Dataset\\Fraud_check.csv')
##Exploring and preparing the data ----
str(fraud)

fraud$risk=ifelse(fraud$Taxable.Income<=30000, "yes", "no")
fraud$risk=as.factor(fraud$risk)
str(fraud)

fraud$Undergrad<-as.numeric(fraud$Undergrad)
fraud$Marital.Status<-as.numeric(fraud$Marital.Status)
fraud$Urban<-as.numeric(fraud$Urban)
fraud$risk<-as.numeric(fraud$risk)
str(fraud)

fraud$Undergrad=as.factor(fraud$Undergrad)
fraud$Marital.Status=as.factor(fraud$Marital.Status)
fraud$Urban=as.factor(fraud$Urban)
fraud$risk=as.factor(fraud$risk)
str(fraud)

fraud$Taxable.Income<-NULL
fraud_rand <- fraud[order(runif(600)), ]
str(fraud_rand)

# split the data frames
fraud_train <- fraud_rand[1:450, ]
fraud_test  <- fraud_rand[451:600, ]

# check the proportion of class variable
prop.table(table(fraud_rand$risk))
prop.table(table(fraud_train$risk))
prop.table(table(fraud_test$risk))

str(fraud_train)
# Step 3: Training a model on the data
library(C50)
fraud_model <- C5.0(fraud_train[,-6], fraud_train$risk)

# Display detailed information about the tree
summary(fraud_model)

# Step 4: Evaluating model performance
# On Training Dataset
train_res <- predict(fraud_model, fraud_train)
train_acc <- mean(fraud_train$risk==train_res)
train_acc

test_res <- predict(fraud_model, fraud_test)
test_acc <- mean(fraud_test$risk==test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_test$risk, test_res, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

