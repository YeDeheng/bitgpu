#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);
require(grid);

pdf(file="range.pdf", height=3.5, width=5)

a<-read.csv('../data/range.dat',header=T)
b<-data.frame(bench=a[,1],splits=a[,4],blocks=a[,3],time=a[,2]);
b<-aggregate(b$time, by=list(bench=b$bench,splits=b$splits), FUN=min)
colnames(b)[3]="time"

c<-read.csv('../data/range_gappa.dat',header=F)
d<-data.frame(bench=c[,1],splits=c[,2],time=c[,3]);

df<-merge(b,d,by=c("bench","splits"))

p <- ggplot(data=df,aes(colour=factor(bench),shape=factor(bench),group=bench)) +
	geom_line(size=0.8,aes(x=df$splits,y=(1e-3*df$time.y/df$time.x)))+
	geom_point(size=3,aes(x=df$splits,y=(1e-3*df$time.y/df$time.x)))+
	scale_colour_hue() + 
	scale_shape_manual(values=1:16)+
	scale_x_log10(limits=c(8,8192),breaks=c(1,8,16,32,64,128,256,512,1024,2048,4096,8192)) + 
	scale_y_log10(limits=c(1,50000), breaks=c(1,10,100,1000,10000,50000)) +
	xlab("Interval Splits") + ylab("Speedup");
	

p +
        theme_bw() +
        theme(
                        legend.position="top",
                        legend.title=element_blank(),
                        legend.background = element_blank(),
                        legend.key=element_blank(),
                        legend.key.width=unit(1.1,"cm"),
                        legend.key.height=unit(0.35,"cm"),
			axis.text.x=element_text(angle=90),
                        legend.text=element_text(size=10)) +
        guides(
                        shape=guide_legend(nrow=3),
                        color=guide_legend(nrow=3)
              ) +
        theme(
                        plot.background = element_blank(),
                        panel.grid.major = element_blank(),
                        panel.grid.minor = element_blank(),
                        panel.border = element_blank()
             ) +
#draws x and y axis line
        theme(axis.line = element_line(color = 'black'));
	
	
