#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);

pdf(file="quality_vs_time.pdf", height=3.5, width=7)

a<-read.csv('../data/raw_asa_poly.dat',header=F)
b<-data.frame(cost=a[,1],time=a[,2])
c<-read.csv('../data/raw_bitslice_poly.dat',header=F)
d<-data.frame(cost=as.integer(c[,1]),time=c[,2]);
d<-d[which.min(d$time),]
mins=apply(b,2,min);
maxs=apply(b,2,max);
b<-data.frame(scale(b,center=mins,scale=maxs-mins))
d<-data.frame(scale(d,center=mins,scale=maxs-mins))

a0<-read.csv('../data/raw_asa_poly6.dat',header=F)
b0<-data.frame(cost=a0[,1],time=a0[,2])
c0<-read.csv('../data/raw_bitslice_poly6.dat',header=F)
d0<-data.frame(cost=as.integer(c0[,1]),time=c0[,2]);
d0<-d0[which.min(d0$time),]
mins=apply(b0,2,min);
maxs=apply(b0,2,max);
b0<-data.frame(scale(b0,center=mins,scale=maxs-mins))
d0<-data.frame(scale(d0,center=mins,scale=maxs-mins))

a1<-read.csv('../data/raw_asa_diode.dat',header=F)
b1<-data.frame(cost=a1[,1],time=a1[,2])
c1<-read.csv('../data/raw_bitslice_diode.dat',header=F)
d1<-data.frame(cost=as.integer(c1[,1]),time=c1[,2]);
d1<-d1[which.min(d1$time),]
mins=apply(b1,2,min);
maxs=apply(b1,2,max);
b1<-data.frame(scale(b1,center=mins,scale=maxs-mins))
d1<-data.frame(scale(d1,center=mins,scale=maxs-mins))

a2<-read.csv('../data/raw_asa_level1_linear.dat',header=F)
b2<-data.frame(cost=a2[,1],time=a2[,2])
c2<-read.csv('../data/raw_bitslice_level1_linear.dat',header=F)
d2<-data.frame(cost=as.integer(c2[,1]),time=c2[,2]);
d2<-d2[which.min(d2$time),]
mins=apply(b2,2,min);
maxs=apply(b2,2,max);
b2<-data.frame(scale(b2,center=mins,scale=maxs-mins))
d2<-data.frame(scale(d2,center=mins,scale=maxs-mins))

a3<-read.csv('../data/raw_asa_level1_satur.dat',header=F)
b3<-data.frame(cost=a3[,1],time=a3[,2])
c3<-read.csv('../data/raw_bitslice_level1_satur.dat',header=F)
d3<-data.frame(cost=as.integer(c3[,1]),time=c3[,2]);
d3<-d3[which.min(d3$time),]
mins=apply(b3,2,min);
maxs=apply(b3,2,max);
b3<-data.frame(scale(b3,center=mins,scale=maxs-mins))
d3<-data.frame(scale(d3,center=mins,scale=maxs-mins))

a4<-read.csv('../data/raw_asa_approx1.dat',header=F)
b4<-data.frame(cost=a4[,1],time=a4[,2])
c4<-read.csv('../data/raw_bitslice_approx1.dat',header=F)
d4<-data.frame(cost=as.integer(c4[,1]),time=c4[,2]);
d4<-d4[which.min(d4$time),]
mins=apply(b4,2,min);
maxs=apply(b4,2,max);
b4<-data.frame(scale(b4,center=mins,scale=maxs-mins))
d4<-data.frame(scale(d4,center=mins,scale=maxs-mins))

df <- rbind(b,b0,b1,b2,b3,b4,d,d0,d1,d2,d3,d4)
df$dataset <- c(rep("poly-cpu",nrow(b)),
		rep("poly6-cpu",nrow(b0)),
		rep("diode-cpu",nrow(b1)),
		rep("level1-linear-cpu",nrow(b2)),
		rep("level1-satur-cpu",nrow(b3)),
		rep("approx1-cpu",nrow(b4)),
		rep("poly-gpu",nrow(d)),
		rep("poly6-gpu",nrow(d0)),
		rep("diode-gpu",nrow(d1)),
		rep("level1-linear-gpu",nrow(d2)),
		rep("level1-satur-gpu",nrow(d3)),
		rep("approx1-gpu",nrow(d4)));

df4=df[grep("gpu",df$dataset),]
df5=df[grep("cpu",df$dataset),]

p <- ggplot() +
	geom_line(data=df5,aes(x=time,y=cost,linetype=df5$dataset),size=0.8) +
	geom_point(data=df4,aes(x=time,y=cost,shape=df4$dataset,colour=df4$dataset),size=4)+
	scale_colour_manual(values=seq(1,12)) + 
	xlab("Normalized Time") + ylab("Normalized Cost") +
	scale_x_log10();
	
p + theme(legend.key=element_blank(),legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10));
