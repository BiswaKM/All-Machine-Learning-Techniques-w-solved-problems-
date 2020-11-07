library(rmarkdown)
library(arules)
library("arulesViz") # for visualizing rules

retail<-read.csv('C:\\Users\\biswa\\Desktop\\360DigiTMG\\15 . Module 15 - Data Mining Unsupervised Learning - Association Rules\\Assignment\\transactions_retail1.csv')
View(retail)
class(retail)
summary(retail) 

# making rules using apriori algorithm 
# Building rules using apriori algorithm

arules <- apriori(retail,parameter=list(support=0.003, confidence = 0.8,minlen=5))
arules
inspect(head(sort(arules,by="lift"))) # to view we use inspect 

# Viewing rules based on lift value

# Overal quality 
head(quality(arules))

# Different Ways of Visualizing Rules

plot(arules)
windows()
plot(arules,method="grouped")
plot(arules,method = "graph") # for good visualization try plotting only few rules
