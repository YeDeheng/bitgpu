#!/usr/bin/Rscript

require(ggplot2);
require(reshape2);
require(grid);

pdf(file="quality_vs_time.pdf", height=3.5, width=5)

a<-read.csv('../data/raw_asa_poly.dat',header=F)
b<-data.frame(cost=a[,1],time=a[,2])
mins=apply(b,2,min);
maxs=apply(b,2,max);
b<-data.frame(scale(b,center=mins,scale=maxs-mins))
c<-read.csv('../data/raw_bitslice_poly.dat',header=F)
d<-data.frame(cost=c[,1],time=c[,2])
d<-data.frame(scale(d,center=mins,scale=maxs-mins))

a0<-read.csv('../data/raw_asa_poly6.dat',header=F)
b0<-data.frame(cost=a0[,1],time=a0[,2])
mins=apply(b0,2,min);
maxs=apply(b0,2,max);
b0<-data.frame(scale(b0,center=mins,scale=maxs-mins))
c0<-read.csv('../data/raw_bitslice_poly6.dat',header=F)
d0<-data.frame(cost=c0[,1],time=c0[,2])
d0<-data.frame(scale(d0,center=mins,scale=maxs-mins))

a1<-read.csv('../data/raw_asa_diode.dat',header=F)
b1<-data.frame(cost=a1[,1],time=a1[,2])
mins=apply(b1,2,min);
maxs=apply(b1,2,max);
b1<-data.frame(scale(b1,center=mins,scale=maxs-mins))
c1<-read.csv('../data/raw_bitslice_diode.dat',header=F)
d1<-data.frame(cost=c1[,1],time=c1[,2])
d1<-data.frame(scale(d1,center=mins,scale=maxs-mins))

a2<-read.csv('../data/raw_asa_level1_linear.dat',header=F)
b2<-data.frame(cost=a2[,1],time=a2[,2])
mins=apply(b2,2,min);
maxs=apply(b2,2,max);
b2<-data.frame(scale(b2,center=mins,scale=maxs-mins))
c2<-read.csv('../data/raw_bitslice_level1_linear.dat',header=F)
d2<-data.frame(cost=c2[,1],time=c2[,2])
d2<-data.frame(scale(d2,center=mins,scale=maxs-mins))

a3<-read.csv('../data/raw_asa_level1_satur.dat',header=F)
b3<-data.frame(cost=a3[,1],time=a3[,2])
mins=apply(b3,2,min);
maxs=apply(b3,2,max);
b3<-data.frame(scale(b3,center=mins,scale=maxs-mins))
c3<-read.csv('../data/raw_bitslice_level1_satur.dat',header=F)
d3<-data.frame(cost=c3[,1],time=c3[,2])
d3<-data.frame(scale(d3,center=mins,scale=maxs-mins))

df <- rbind(b,b0,b1,b2,b3)
df$dataset <- c(rep("poly-asa",nrow(b)),
		rep("poly6-asa",nrow(b0)),
		rep("diode-asa",nrow(b1)),
		rep("level1-linear-asa",nrow(b2)),
		rep("level1-satur-asa",nrow(b3)));

df2 <- rbind(d,d0,d1,d2,d3)
df2$dataset2 <- c(rep("poly-bitslice",nrow(d)),
		rep("poly6-bitslice",nrow(d0)),
		rep("diode-bitslice",nrow(d1)),
		rep("level1-linear-bitslice",nrow(d2)),
		rep("level1-satur-bitslice",nrow(d3)));

df3<-data.frame(aggregate(df2$time,list(cost=df2$cost,dataset2=df2$dataset2),min));
colnames(df3)[3] <- "time";

p <- ggplot() +
	geom_point(data=df3,aes(x=time,y=cost,colour=dataset2,shape=dataset2),size=2)+
	scale_colour_hue(breaks=c("poly-asa",NA,"poly6-asa",NA,"diode-asa",NA,"level1-linear-asa",NA,"level1-satur-asa",NA))+
	geom_line(data=df,aes(x=time,y=cost,colour=dataset,shape=NULL),size=0.1) +
	xlab("Time (s)") + ylab("Cost (LUTs)") +
	theme_bw();

p +
        theme_bw() +
        theme(
                        legend.position="right",
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


