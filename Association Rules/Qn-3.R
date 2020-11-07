library(rmarkdown)
library(arules)
library("arulesViz") # for visualizing rules

phone<-read.csv('C:\\Users\\biswa\\Desktop\\360DigiTMG\\15 . Module 15 - Data Mining Unsupervised Learning - Association Rules\\Assignment\\myphonedata.csv')
View(phone)
class(phone)
phone$V1<-NULL
phone$V2<-NULL
phone$V3<-NULL
summary(phone) 

# making rules using apriori algorithm 
# Building rules using apriori algorithm

arules <- apriori(as.matrix(phone),parameter=list(support=0.002, confidence = 0.5,minlen=2))
arules
inspect(head(sort(arules,by="lift"))) # to view we use inspect 

# Viewing rules based on lift value

# Overal quality 
head(quality(arules))

# Different Ways of Visualizing Rules

plot(arules)
windows()
plot(arules,method="grouped")
plot(arules[1:20],method = "graph") # for good visualization try plotting only few rules
