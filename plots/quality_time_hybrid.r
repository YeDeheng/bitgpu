#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);

pdf(file="quality_vs_time_hybrid.pdf", height=3.5, width=7)

a<-read.csv('../data/hybrid.dat',header=T,row.names=1)
b<-data.frame(a)

p <- ggplot() +
	geom_line(data=df5,aes(x=time,y=cost,linetype=df5$dataset),size=0.8) +
	geom_point(data=df4,aes(x=time,y=cost,shape=df4$dataset,colour=df4$dataset),size=4)+
	scale_colour_manual(values=seq(1,12)) + 
	xlab("Benchmark") + ylab("Speedup") +
	scale_x_log10();
	
p + theme(legend.key=element_blank(),legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10));
