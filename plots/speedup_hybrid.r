#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);
require(grid);

pdf(file="speedup_hybrid.pdf", height=3.5, width=7)

a<-read.csv('../data/hybrid_speedup.dat',header=T,row.names=1, check.names=F)
m<-as.matrix(a)
m[m>1000]<-0
m<-m[,seq(1,4)]
a<-melt(m)

p <- ggplot(data=a,aes(group=a$Var2,fill=factor(a$Var2))) +
	geom_bar(position=position_dodge(),stat="identity",aes(x=a$Var1,y=a$value),width=0.5) +
	geom_bar(colour='black',position=position_dodge(),stat="identity",aes(x=a$Var1,y=a$value),width=0.5,show_guide=F) +
	scale_fill_grey() +
	theme_bw() + scale_y_log10() +
	xlab("Benchmark") + ylab("Speedup");
	

p +
        theme_bw() +
        theme(
                        legend.direction="horizontal",
                        legend.position=c(0.4,0.95),
                        legend.title=element_blank(),
                        legend.background = element_blank(),
                        legend.key=element_blank(),
                        legend.key.width=unit(1.1,"cm"),
                        legend.key.height=unit(0.35,"cm"),
			legend.position=c(0.35,0.9),
			axis.text.x=element_text(angle=90,size=15),
                        legend.text=element_text(size=15)) +
        guides(
                        shape=guide_legend(ncol=1),
                        color=guide_legend(ncol=1)
              ) +
        theme(
                        plot.background = element_blank(),
                        panel.grid.major = element_blank(),
                        panel.grid.minor = element_blank(),
                        panel.border = element_blank()
             ) +
#draws x and y axis line
        theme(axis.line = element_line(color = 'black'));


