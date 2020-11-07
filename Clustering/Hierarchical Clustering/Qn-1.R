library(readxl)
mydata=read_excel('C:/Users/Biswa/Desktop/360DigiTMG/12 . Module 12/Assignment/Data Set/EastWestAirlines.xlsx',sheet=2)
mydata=mydata[,2:12]


normalized_data <- scale(mydata)
d <- dist(normalized_data, method = "euclidean") # distance matrix
fit <- hclust(d, method="complete")
plot(fit) # display dendrogram
plot(fit, hang=-1)
groups <- cutree(fit, k=3) # cut tree into 3 clusters

rect.hclust(fit, k=3, border="red")

membership<-as.matrix(groups)

final <- data.frame(mydata, membership)

final1 <- final[,c(ncol(final),1:(ncol(final)-1))]

