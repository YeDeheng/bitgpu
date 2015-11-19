#!/usr/bin/Rscript

require(lattice);
require(reshape2);
require(ggplot2);

a0<-read.csv('raw_kernel_time.csv',header=T);
a1<-read.csv('raw_kernel_time_openmp.csv',header=T);
pdf(file="raw_speedup.pdf", height=3.5, width=7.5);

b0<-data.frame(operation=a0$operation,model=a0$model,time=a0$runtime);
b1<-data.frame(operation=a1$operation,model=a1$model,time=a1$runtime);
c0<-aggregate(b0[3],b0[1:2],min);
c1<-aggregate(b1[3],b1[1:2],min);

d<-data.frame(operation=c0$operation,model=c0$model,speedup=c1$time/c0$time);

e<-melt(d,id.vars=c("operation","model"));
e$operation=e$operation+1;
 
p <- ggplot(data=e,aes(x=operation,y=value,group=model,fill=factor(model)))+geom_bar(width=0.5,stat="identity",position=position_dodge()) +
scale_x_discrete(breaks=seq(0,6,1),labels=c('','add','sub','mult','div','exp','log'))+
scale_fill_manual(labels=c('range','error','area'),values=c('#40E0D0','#FF6347',"#C7C7C7"))+
xlab('Arithmetic Operation') + ylab('GPU Speedup');

p + theme_bw() + theme(legend.position=c(0.85,0.8), legend.title=element_blank(),legend.background = element_blank(),legend.text=element_text(size=10)); 

