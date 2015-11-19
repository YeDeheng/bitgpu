#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);

pdf(file="speedup.pdf", height=3.5, width=7)

a<-read.csv('../data/raw_asa_poly.dat',header=F)
b<-data.frame(cost=a[,1],time=a[,2])
c<-read.csv('../data/raw_bitslice_poly.dat',header=F)
d<-data.frame(cost=as.integer(c[,1]),time=c[,2]);
d<-d[which.min(d$time),]
b<-tail(b,1);
e<-b[,2]/d[,2]

a0<-read.csv('../data/raw_asa_poly6.dat',header=F)
b0<-data.frame(cost=a0[,1],time=a0[,2])
c0<-read.csv('../data/raw_bitslice_poly6.dat',header=F)
d0<-data.frame(cost=as.integer(c0[,1]),time=c0[,2]);
d0<-d0[which.min(d0$time),]
b0<-tail(b0,1);
e0<-b0[,2]/d0[,2]

a1<-read.csv('../data/raw_asa_diode.dat',header=F)
b1<-data.frame(cost=a1[,1],time=a1[,2])
c1<-read.csv('../data/raw_bitslice_diode.dat',header=F)
d1<-data.frame(cost=as.integer(c1[,1]),time=c1[,2]);
d1<-d1[which.min(d1$time),]
b1<-tail(b1,1);
e1<-b1[,2]/d1[,2]

a2<-read.csv('../data/raw_asa_level1_linear.dat',header=F)
b2<-data.frame(cost=a2[,1],time=a2[,2])
c2<-read.csv('../data/raw_bitslice_level1_linear.dat',header=F)
d2<-data.frame(cost=as.integer(c2[,1]),time=c2[,2]);
d2<-d2[which.min(d2$time),]
b2<-tail(b2,1);
e2<-b2[,2]/d2[,2]

a3<-read.csv('../data/raw_asa_level1_satur.dat',header=F)
b3<-data.frame(cost=a3[,1],time=a3[,2])
c3<-read.csv('../data/raw_bitslice_level1_satur.dat',header=F)
d3<-data.frame(cost=as.integer(c3[,1]),time=c3[,2]);
d3<-d3[which.min(d3$time),]
b3<-tail(b3,1);
e3<-b3[,2]/d3[,2]

a4<-read.csv('../data/raw_asa_approx1.dat',header=F)
b4<-data.frame(cost=a4[,1],time=a4[,2])
c4<-read.csv('../data/raw_bitslice_approx1.dat',header=F)
d4<-data.frame(cost=as.integer(c4[,1]),time=c4[,2]);
d4<-d4[which.min(d4$time),]
b4<-tail(b4,1);
e4<-b4[,2]/d4[,2]

df <- rbind(e,e0,e1,e2,e3,e4)
df1 <- data.frame(df);
df1$dataset <- c("poly",
		"poly6",
		"diode",
		"level1-linear",
		"level1-satur",
		"approx1");

p <- ggplot(data=df1,aes(fill=df1$dataset)) +
	geom_bar(width=0.5,stat="identity",aes(x=df1$dataset,y=df1$df),position=position_dodge(), show_guide=F) +
	scale_fill_hue() + 
	xlab("Benchmarks") + ylab("Speedup");
	
p + theme(legend.key=element_blank(),legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10));
