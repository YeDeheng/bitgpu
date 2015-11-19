#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);
#require(doBy);
require(data.table);

a<-read.csv('../data/quality_time_asa.dat',header=T)
a$cost<-as.double(a$cost);

b<-data.table(a);
b[,time:=(time-min(time))/(max(time)-min(time)), by="bench"]
b[,cost:=(cost-min(cost))/(max(cost)-min(cost)), by="bench"]

b$time[b$time<=1e-3]<-1e-3

c<-as.data.frame(b);
#c<-c[c$bench=="approx1"|c$bench=="approx2",]

pdf(file="quality_time_asa.pdf", height=3.5, width=5)

p <- ggplot(data=c, aes(x=time,y=cost,group=bench,colour=bench,fill=bench)) + 
	geom_area(show_guide=F,alpha=0.1) + 
#	geom_area(alpha=0,position="stack",show_guide=F,colour="black") +  
	ylim(0,1) + xlim(0,1) + scale_x_log10() +
	geom_line(aes(x=time,y=cost),size=1) +
	xlab("Normalized Time") + ylab("Normalize Cost") +
#	scale_x_log10() + 
#	scale_y_log10() + 
#	scale_colour_grey() + 
	scale_colour_hue() + 
	theme_bw();

p + theme(legend.key=element_blank(),legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10));
