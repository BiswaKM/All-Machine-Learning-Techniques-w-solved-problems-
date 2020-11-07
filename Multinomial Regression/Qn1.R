require('mlogit')
require('nnet')

mdata=read.csv('C:\\Users\\Biswa\\Desktop\\360DigiTMG\\10 . Module 10 - Multinomial Regression\\Assignment\\Dataset\\mdata.csv')
head(mdata)
tail(mdata)
colnames(mdata)
mdata$X<-NULL
mdata$id<-NULL

table(mdata$prog) # tabular representation of the Y categories
mdata$female<-as.numeric(mdata$female)
mdata$ses<-as.numeric(mdata$ses)
mdata$schtyp<-as.numeric(mdata$schtyp)
mdata$honors<-as.numeric(mdata$honors)

mdata.prog <- multinom(prog ~ female+ses+schtyp+read+write+math+science+honors, data=mdata)
summary(mdata.prog)

mdata$prog  <- relevel(mdata$prog, ref= 'general')  # change the baseline level

##### Significance of Regression Coefficients###
z <- summary(mdata.prog)$coefficients / summary(mdata.prog)$standard.errors
p_value <- (1-pnorm(abs(z),0,1))*2

summary(mdata.prog)$coefficients
p_value

# odds ratio 
exp(coef(mdata.prog))

# predict probabilities
prob <- fitted(mdata.prog)
prob

# Find the accuracy of the model

class(prob)
prob <- data.frame(prob)
prob["pred"] <- NULL

# Custom function that returns the predicted value based on probability
get_names <- function(i){
  return (names(which.max(i)))
}

pred_name <- apply(prob,1,get_names)
prob$pred <- pred_name

# Confusion matrix
table(pred_name,mdata$prog)

# confusion matrix visualization
barplot(table(pred_name,mdata$prog),beside = T,col=c("red","lightgreen","blue","orange"),legend=c("bus","car","carpool","rail"),main = "Predicted(X-axis) - Legends(Actual)",ylab ="count")


# Accuracy 
mean(pred_name==mdata$prog)
