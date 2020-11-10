sales=read.csv('C:\\Users\\Biswa\\Desktop\\360DigiTMG\\19 . Module 19 - Decision Tree - Random Forest\\Assignment\\Dataset\\Company_Data.csv')
##Exploring and preparing the data ----
str(sales)

sales$Outcome=ifelse(sales$Sales<9, "No", "Yes")
sales$Outcome=as.factor(sales$Outcome)
str(sales)
sales$Sales<-NULL
sales_rand <- sales[order(runif(400)), ]
str(sales_rand)

sales$ShelveLoc<-as.numeric(sales$ShelveLoc)
sales$Urban<-as.numeric(sales$Urban)
sales$US<-as.numeric(sales$US)
sales$Outcome<-as.numeric(sales$Outcome)
str(sales)

sales$ShelveLoc=as.factor(sales$ShelveLoc)
sales$Urban=as.factor(sales$Urban)
sales$US=as.factor(sales$US)
sales$Outcome=as.factor(sales$Outcome)
str(sales)

# split the data frames
sales_train <- sales_rand[1:300, ]
sales_test  <- sales_rand[301:400, ]

# check the proportion of class variable
prop.table(table(sales_rand$Outcome))
prop.table(table(sales_train$Outcome))
prop.table(table(sales_test$Outcome))

str(sales_train)
# Step 3: Training a model on the data
library(C50)
sales_model <- C5.0(sales_train[,-11], sales_train$Outcome)

# Display detailed information about the tree
summary(sales_model)

# Step 4: Evaluating model performance
# On Training Dataset
train_res <- predict(sales_model, sales_train)
train_acc <- mean(sales_train$Outcome==train_res)
train_acc

test_res <- predict(sales_model, sales_test)
test_acc <- mean(sales_test$Outcome==test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(sales_test$Outcome, test_res, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

