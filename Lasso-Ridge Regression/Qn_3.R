toyota <- read.csv('C:/Users/biswa/Desktop/360DigiTMG/8 . Module 8 - Lasso-Ridge Regression/Assignment/Datsets/ToyotaCorolla.csv') # loading the data 
toyota$Id<-NULL
toyota$Model<-NULL
attach(toyota)
str(toyota)
# Linear regression
m3 <- lm(Price ~ .,data =toyota )
summary(m3)

predprice <- predict(m3, data=comp)
MSE3 <- mean(m3$residuals)^2

# Residual plot
plot(toyota$Price, predprice)

barplot(sort(m3$coefficients), ylim=c(-0.5, 5))
# Regularization methods
#########################
str(comp)
# Converting the data into compatible format in which model accepts 
comp_x <- model.matrix(Price~.-1,data=toyota)
comp_y <- toyota$Price
library(glmnet)

# Lambda is the hyperparameter to tune the ridge regression

# glmnet automatically selects the range of λ values
# setting lamda as 10^10 till 10^-2
lambda <- 10^seq(10, -2, length = 50)

# For ridge alpha = 0

# Note: glmnet() function standardizes the variables to get them on to same scale by default. 

ridge_reg <- glmnet(comp_x,comp_y,alpha=0,lambda=lambda)
summary(ridge_reg)

# ----------
# Below graph shows how the coefficients vary with change in lambda
# With increase in lambda the coefficients value converges to 0 
plot(ridge_reg,xvar="lambda",label=T)

# ridge regression coefficients, stored in a matrix 
dim(coef(ridge_reg))
plot(ridge_reg)

ridge_reg$lambda[1]  #Display 1st lambda value
coef(ridge_reg)[,1] # Display coefficients associated with 50th lambda value
sqrt(sum(coef(ridge_reg)[-1,1]^2)) # Calculate L2 norm

ridge_reg$lambda[50]
coef(ridge_reg)[,50] 
sqrt(sum(coef(ridge_reg)[-1,41]^2)) # Calculate L2 norm
# Larger L2 norm for smaller values of lamda

# ------------


#######
# Partitioning Data into training set and testing set

train <- toyota[1:1149,]
test <- toyota[1150:1436,]

x_train <- model.matrix(Price~.-1,data=train)
y_train <- train$Price

x_test <- model.matrix(Price~.-1,data=test)
y_test <- test$Price

### Ridge Regression

ridge_mod = glmnet(x_train, y_train, alpha=0, lambda = lambda)
plot(ridge_mod)

ridge_pred = predict(ridge_mod, s = -2, newx = x_test)
mean((ridge_pred - y_test)^2)

# Fit ridge regression model on training data
cv.out = cv.glmnet(x_train, y_train, alpha = 0) 

# Select lamda that minimizes training MSE
bestlam = cv.out$lambda.min  
bestlam

# Draw plot of training MSE as a function of lambda
plot(cv.out) 

# predicting on test data with best lambda
ridge_pred1 = predict(ridge_mod, s = bestlam, newx = x_test)
mean((ridge_pred1 - y_test)^2) # Calculate test MSE


###
# LASSO Regression

# Fit lasso model on training data
lasso_mod = glmnet(x_train,y_train, alpha = 1, lambda = lambda)

plot(lasso_mod)    # Draw plot of coefficients

cv.out = cv.glmnet(x_train, y_train, alpha = 1) # Fit lasso model on training data

plot(cv.out) # Draw plot of training MSE as a function of lambda

bestlam_lasso = cv.out$lambda.min # Select lamda that minimizes training MSE
bestlam_lasso
# Use best lambda to predict test data
lasso_pred = predict(lasso_mod, s = bestlam_lasso, newx = x_test)

mean((lasso_pred - y_test)^2) # Calculate test MSE

# Fit lasso model on full dataset
out = glmnet(comp_x, comp_y, alpha = 1, lambda = lambda) 

# Display coefficients using lambda chosen by CV
lasso_coef = predict(out, type = "coefficients", s = bestlam)[1:5,] 
lasso_coef

