input <- read.csv(file.choose())
mydata <- input[,c(2:14)]

## the first column in mydata has university names
data <- mydata
attach(data)

pcaObj<-princomp(data, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev*pcaObj$sdev)*100/(sum(pcaObj$sdev*pcaObj$sdev)),type="b")

pcaObj$scores


# Top 3 pca scores 
final<-cbind(input[,1],pcaObj$scores[,1:3])

final<-final[,2:4]

#Heirarchical clustering
normalized_data <- scale(final)
d <- dist(normalized_data, method = "euclidean") # distance matrix
fit <- hclust(d, method="complete")
plot(fit) # display dendrogram
plot(fit, hang=-1)
groups <- cutree(fit, k=3) # cut tree into 3 clusters

rect.hclust(fit, k=3, border="red")

Hierarchical<-as.matrix(groups)

final2 <- data.frame(mydata, Hierarchical)

#K-Means clustering

wss = (nrow(normalized_data)-1)*sum(apply(normalized_data, 2, var))		 # Determine number of clusters by scree-plot 
for (i in 1:8) wss[i] = sum(kmeans(normalized_data, centers=i)$withinss)
plot(1:8, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")   # Look for an "elbow" in the scree plot #
title(sub = "K-Means Clustering Scree-Plot")

kmeans<- kmeans(normalized_data, 3)
str(kmeans)
final2<- data.frame(final2, kmeans$cluster) # append cluster membership
final2
aggregate(final2, by=list(kmeans$cluster), FUN=mean)
