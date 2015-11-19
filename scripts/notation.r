#!/usr/bin/Rscript

args<-commandArgs(TRUE);
filei=args[1]
fileo=args[2]

a<-read.csv(filei,header=FALSE)
#strip out ;
a$V5<-gsub(";","",a$V4)
a$V5<-sapply(a$V5, as.numeric)
a$V4<-a$V5
a$V5<-NULL
#convert format
a$V4<-format(a$V4,scientific=FALSE)
a$V3<-format(a$V3,scientific=FALSE)

#fix rounding
b<-a[a$V1=='LD',]
c<-a[a$V1!='LD',]
c$V3<-sapply(c$V3, as.numeric)
c$V4<-sapply(c$V4, as.numeric)
c$V3<-round(c$V3,digits=0)
c$V4<-round(c$V4,digits=0)

d<-rbind(b,c)
d$V4<-paste(d$V4,";")
write.table(d,fileo,quote=FALSE,row.names=FALSE,col.names=FALSE,sep=",");
