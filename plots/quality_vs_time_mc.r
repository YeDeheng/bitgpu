#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);

pdf(file="quality_vs_time_mc.pdf", height=3.5, width=5)

a<-read.csv('../data/quality_time_mc.dat',header=F);
b<-data.frame(bench=a[,1],seed=a[,2],samples=a[,3],cost=a[,4],time=a[,5]);
b<-subset(b,b$seed==1) ; 
#b$samples==1000000 & b$seed==1);

p <- ggplot(data=b,aes(group=bench,shape=bench)) +
	geom_point(aes(x=time,y=cost),size=2)+
	geom_line(aes(x=time,y=cost),size=0.3) +
	scale_colour_hue() +
	scale_x_log10() + #scale_y_log10() +
	xlab("Time (s)") + ylab("Cost (LUTs)") +
	theme_bw();
#	scale_x_log10() + scale_y_log10() +
	
p + theme(legend.key=element_blank(),legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10));
