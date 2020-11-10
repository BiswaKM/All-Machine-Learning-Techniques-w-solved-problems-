# Read the dataset
df <- read.csv(file.choose())

table(df$Class.variable)

round(prop.table(table(df$Class.variable))*100,1)
df$Class.variable<-as.numeric(df$Class.variable)
df$Class.variable<-as.factor(df$Class.variable)
str(df)
#Create a function to normalize the data
norm <- function(x){ 
  return((x-min(x))/(max(x)-min(x)))
}

#Apply the normalization function to wbcd dataset
df_n <- as.data.frame(lapply(df[2:8], norm))

# create training and test data
df_train <- df_n[1:576, ]
df_test <- df_n[577:768, ]

# create labels for training and test data
df_train_labels <- df[1:576, 9]
df_test_labels <- df[577:768, 9]


# Building a random forest model on training data 
library(randomForest)

df_forest <- randomForest(df_train_labels~.,data=df_train,importance=TRUE)
plot(df_forest)

# Train Data Accuracy
train_acc <- mean(df_train_labels==predict(df_forest))
train_acc

# Test Data Accuracy
test_acc <- mean(df_test_labels==predict(df_forest, newdata=df_test))
test_acc

varImpPlot(df_forest)
