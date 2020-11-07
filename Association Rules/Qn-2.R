library(rmarkdown)
library(arules)
library("arulesViz") # for visualizing rules

movies<-read.csv('C:\\Users\\biswa\\Desktop\\360DigiTMG\\15 . Module 15 - Data Mining Unsupervised Learning - Association Rules\\Assignment\\my_movies.csv')
View(movies)
class(movies)
movies$V1<-NULL
movies$V2<-NULL
movies$V3<-NULL
movies$V4<-NULL
movies$V5<-NULL
summary(movies) 

# making rules using apriori algorithm 
# Building rules using apriori algorithm

arules <- apriori(as.matrix(movies),parameter=list(support=0.02, confidence = 0.5,minlen=3))
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
