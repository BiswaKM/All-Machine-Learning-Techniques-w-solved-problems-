sales=read.csv(file.choose())
attach(sales)
table1<-table(Weekdays)
table1
table2<-table(Weekend)
table2
prop.test(x=c(113,167),n=c(400,400),conf.level = 0.95,alternative = "two.sided")
prop.test(x=c(113,167),n=c(400,400),conf.level = 0.95,alternative = "greater")
