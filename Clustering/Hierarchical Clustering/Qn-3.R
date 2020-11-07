mydata=read.csv(file.choose())
mydata[1]=NULL

normalized_data <- scale(mydata)
d <- dist(normalized_data, method = "euclidean") # distance matrix
fit <- hclust(d, method="complete")
plot(fit) # display dendrogram
plot(fit, hang=-1)
groups <- cutree(fit, k=5) # cut tree into 3 clusters

rect.hclust(fit, k=5, border="red")

membership<-as.matrix(groups)

final <- data.frame(mydata, membership)

final1 <- final[,c(ncol(final),1:(ncol(final)-1))]

?write.xlsx

write.csv(final1, file="Qn-3.csv")

getwd()

