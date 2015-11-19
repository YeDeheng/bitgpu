#!/usr/bin/Rscript

require(ggplot2)

m<-read.csv('quality_time.csv',header=T)

pdf(file="quality_time.pdf", height=3.5, width=5);


bf_data <- data.frame(x=m$bf_time,y=m$bf_cost)
mc_min_data <- data.frame(x=m$mc_time,y=m$mc_MIN)
mc_max_data <- data.frame(x=m$mc_time,y=m$mc_MAX)
mc_mean_data <- data.frame(x=m$mc_time,y=m$mc_GEOMEAN)
asa_data <- data.frame(x=m$asa_time,y=m$asa_best_cost)

df <- rbind(bf_data,mc_min_data,mc_max_data,mc_mean_data,asa_data)
df$dataset <- c(rep("A", nrow(bf_data)), rep("B1", nrow(mc_min_data)), rep("B2", nrow(mc_max_data)), rep("B3", nrow(mc_mean_data)), rep("C", nrow(asa_data)))

mc_max_data_perm<-mc_max_data[order(-mc_max_data$x),]
df1 <- rbind(mc_max_data_perm,mc_min_data)
df1$dataset <- c(rep("B1", nrow(mc_max_data_perm)), rep("B2", nrow(mc_min_data)))
df2<-na.omit(df1);
df3<-data.frame(x=df2$x,y=df2$y,dataset=df2$dataset)

p <- ggplot(data=df)+
	geom_polygon(data=df3,aes(x=x,y=y),alpha=0.15)+
	geom_line(aes(x=x,y=y,col=dataset,shape=dataset))+
	geom_point(aes(x=x,y=y,col=dataset,shape=dataset))+
	scale_colour_hue(name="",labels=c("Brute-Force GPU","Monte-Carlo GPU (Min)","Monte-Carlo GPU (Max)","Monte-Carlo GPU (Mean)","ASA CPU"))+
	scale_shape_manual(name="",labels=c("Brute-Force GPU","Monte-Carlo GPU (Min)","Monte-Carlo GPU (Max)","Monte-Carlo GPU (Mean)","ASA CPU"),values=c(19,NA,NA,NA,NA))+
	xlab("Time(ms)")+ylab("Quality(LUTs)")+scale_x_log10(limits=c(0.2,10));

p + theme(legend.position=c(0.75,0.7), legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10));

