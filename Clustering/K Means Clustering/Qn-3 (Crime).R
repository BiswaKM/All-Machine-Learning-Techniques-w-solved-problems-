input <-read.csv(file.choose()) 
mydata=input
mydata[1]=NULL
normalized_data <- scale(mydata)

#elbow curve & k ~ sqrt(n/2) to decide the k value

wss = (nrow(normalized_data)-1)*sum(apply(normalized_data, 2, var))		 # Determine number of clusters by scree-plot 
for (i in 1:8) wss[i] = sum(kmeans(normalized_data, centers=i)$withinss)
plot(1:8, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")   # Look for an "elbow" in the scree plot #
title(sub = "K-Means Clustering Scree-Plot")

fit <- kmeans(normalized_data, 3)
str(fit)
final2<- data.frame(mydata, fit$cluster) # append cluster membership
final2
final3 <- final2[,c(ncol(final2),1:(ncol(final2)-1))]
aggregate(mydata, by=list(fit$cluster), FUN=mean)

